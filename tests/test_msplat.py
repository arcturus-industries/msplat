"""msplat test suite."""

import pytest
import numpy as np
import tempfile
import os
import platform
import struct

GARDEN = os.path.join(os.path.dirname(__file__), "..", "datasets", "mipnerf360", "garden")
HAS_GARDEN = os.path.isdir(GARDEN)


# ── Import tests ─────────────────────────────────────────────────────────────


def test_import():
    import msplat
    assert hasattr(msplat, "GaussianTrainer")
    assert hasattr(msplat, "GaussianRenderer")
    assert hasattr(msplat, "TrainingConfig")
    assert hasattr(msplat, "Dataset")
    assert hasattr(msplat, "load_dataset")


def _write_tiny_gaussian_ply(path):
    c0 = 0.28209479177387814
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            "element vertex 1",
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property float f_dc_0",
            "property float f_dc_1",
            "property float f_dc_2",
            "property float opacity",
            "property float scale_0",
            "property float scale_1",
            "property float scale_2",
            "property float rot_0",
            "property float rot_1",
            "property float rot_2",
            "property float rot_3",
            "end_header",
            "",
        ]
    ).encode("ascii")
    row = struct.pack(
        "<17f",
        0.0,
        0.0,
        -2.0,
        0.0,
        0.0,
        0.0,
        (1.0 - 0.5) / c0,
        (0.2 - 0.5) / c0,
        (0.1 - 0.5) / c0,
        5.0,
        np.log(0.25),
        np.log(0.25),
        np.log(0.25),
        1.0,
        0.0,
        0.0,
        0.0,
    )
    with open(path, "wb") as f:
        f.write(header)
        f.write(row)


def _write_overflow_gaussian_ply(path, count=3000):
    c0 = 0.28209479177387814
    props = [
        "x",
        "y",
        "z",
        "nx",
        "ny",
        "nz",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    ]
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {count}",
    ]
    header += [f"property float {p}" for p in props]
    header += ["end_header", ""]
    rows = []
    for i in range(count):
        rows.append(
            struct.pack(
                "<17f",
                0.0,
                0.0,
                -2.0 - i * 1e-6,
                0.0,
                0.0,
                0.0,
                (0.8 - 0.5) / c0,
                (0.3 - 0.5) / c0,
                (0.1 - 0.5) / c0,
                -5.0,
                np.log(0.01),
                np.log(0.01),
                np.log(0.01),
                1.0,
                0.0,
                0.0,
                0.0,
            )
        )
    with open(path, "wb") as f:
        f.write("\n".join(header).encode("ascii"))
        for row in rows:
            f.write(row)


def test_training_config_defaults():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    assert cfg.iterations == 30000
    assert cfg.sh_degree == 3
    assert cfg.ssim_weight == pytest.approx(0.2)
    assert cfg.refine_every == 100
    assert cfg.warmup_length == 500


def test_training_config_custom():
    from msplat import TrainingConfig

    cfg = TrainingConfig(iterations=100, sh_degree=1, ssim_weight=0.0)
    assert cfg.iterations == 100
    assert cfg.sh_degree == 1
    assert cfg.ssim_weight == 0.0


def test_training_config_mutable():
    from msplat import TrainingConfig

    cfg = TrainingConfig()
    cfg.iterations = 500
    assert cfg.iterations == 500


# ── Dataset tests ────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    assert ds.num_train > 0
    assert ds.num_test > 0
    assert ds.num_train + ds.num_test > 100


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_load_dataset_no_eval():
    from msplat import Dataset

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=False)
    assert ds.num_train > 0
    assert ds.num_test == 0


# ── Training tests ───────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_train_short():
    """Train 50 steps at 4x downscale — verify it runs without error."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    steps_seen = []
    trainer.train(lambda s: steps_seen.append(s.iteration), callback_every=10)

    assert trainer.iteration == 50
    assert trainer.splat_count > 100000
    assert steps_seen == [10, 20, 30, 40, 50]


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_step_by_step():
    """Manual step loop works."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        stats = trainer.step()

    assert stats.iteration == 10
    assert stats.splat_count > 0
    assert stats.ms_per_step > 0


# ── Render tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_render():
    """Render produces valid image array."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer, sync

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    img = trainer.render(0)
    assert isinstance(img, np.ndarray)
    assert img.dtype == np.float32
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[0] > 0 and img.shape[1] > 0
    # Values should be in [0, 1] range (approximately)
    assert img.min() >= -0.1
    assert img.max() <= 1.5


@pytest.mark.skipif(platform.system() != "Darwin", reason="GaussianRenderer requires Metal")
def test_gaussian_renderer_from_ply():
    """Render-only PLY API accepts explicit pose and intrinsics."""
    from msplat import GaussianRenderer

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        _write_tiny_gaussian_ply(path)
        renderer = GaussianRenderer(path, bg_color=[0.0, 0.0, 0.0])
        pose = np.eye(4, dtype=np.float32)
        img = renderer.render(
            pose,
            64,
            64,
            48.0,
            48.0,
            32.0,
            32.0,
            max_sh_degree=0,
        )

        assert renderer.splat_count == 1
        assert isinstance(img, np.ndarray)
        assert img.dtype == np.float32
        assert img.shape == (64, 64, 3)
        assert np.isfinite(img).all()
        assert img.max() > 0.0
    finally:
        os.unlink(path)




@pytest.mark.skipif(platform.system() != "Darwin", reason="GaussianRenderer requires Metal")
def test_gaussian_renderer_render_depth_from_ply():
    """Render-only PLY API can produce expected depth and alpha."""
    from msplat import GaussianRenderer

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        _write_tiny_gaussian_ply(path)
        renderer = GaussianRenderer(path, bg_color=[0.0, 0.0, 0.0], overflow_mode="exact")
        pose = np.eye(4, dtype=np.float32)
        depth, alpha = renderer.render_depth(
            pose,
            64,
            64,
            48.0,
            48.0,
            32.0,
            32.0,
            max_sh_degree=0,
        )

        assert depth.dtype == np.float32
        assert alpha.dtype == np.float32
        assert depth.shape == (64, 64)
        assert alpha.shape == (64, 64)
        assert np.isfinite(depth).all()
        assert np.isfinite(alpha).all()
        assert alpha.max() > 0.0
        assert depth[alpha > 0.05].max() > 0.0
    finally:
        os.unlink(path)


@pytest.mark.skipif(platform.system() != "Darwin", reason="GaussianRenderer requires Metal")
def test_gaussian_renderer_exact_overflow_mode_repairs_tiles():
    """Exact overflow mode rerenders tiles that exceed the fast-path bin limit."""
    from msplat import (
        GaussianRenderer,
        last_overflow_max_tile_count,
        last_overflow_tile_count,
        overflow_fallback_count,
        reset_overflow_fallback_count,
    )

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        _write_overflow_gaussian_ply(path, count=3000)
        reset_overflow_fallback_count()
        renderer = GaussianRenderer(path, bg_color=[0.0, 0.0, 0.0], overflow_mode="exact")
        pose = np.eye(4, dtype=np.float32)
        img = renderer.render(
            pose,
            64,
            64,
            48.0,
            48.0,
            32.0,
            32.0,
            max_sh_degree=0,
        )

        assert img.shape == (64, 64, 3)
        assert np.isfinite(img).all()
        assert overflow_fallback_count() >= 1
        assert last_overflow_tile_count() >= 1
        assert last_overflow_max_tile_count() >= 3000
    finally:
        os.unlink(path)


@pytest.mark.skipif(platform.system() != "Darwin", reason="GaussianRenderer requires Metal")
def test_gaussian_renderer_exact_depth_does_not_pollute_rgb_state():
    """Depth exact-overflow fallback must not corrupt following RGB renders."""
    from msplat import GaussianRenderer, reset_overflow_fallback_count

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        _write_overflow_gaussian_ply(path, count=3000)
        reset_overflow_fallback_count()
        renderer = GaussianRenderer(path, bg_color=[0.0, 0.0, 0.0], overflow_mode="exact")
        pose = np.eye(4, dtype=np.float32)
        rgb_args = (pose, 64, 64, 48.0, 48.0, 32.0, 32.0)
        depth_args = (pose, 32, 32, 24.0, 24.0, 16.0, 16.0)

        rgb_before = renderer.render(*rgb_args, max_sh_degree=0)
        depth, alpha = renderer.render_depth(*depth_args, max_sh_degree=0)
        rgb_after = renderer.render(*rgb_args, max_sh_degree=0)

        assert np.isfinite(depth).all()
        assert np.isfinite(alpha).all()
        assert np.isfinite(rgb_after).all()
        np.testing.assert_allclose(rgb_after, rgb_before, rtol=0.0, atol=1e-6)
    finally:
        os.unlink(path)


# ── Export tests ─────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_ply():
    """PLY export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as f:
        path = f.name

    try:
        trainer.export_ply(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000  # non-trivial file
    finally:
        os.unlink(path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_export_splat():
    """Splat export creates a valid file."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=10, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(10):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".splat", delete=False) as f:
        path = f.name

    try:
        trainer.export_splat(path)
        assert os.path.exists(path)
        size = os.path.getsize(path)
        assert size > 1000
    finally:
        os.unlink(path)


# ── Eval tests ───────────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_evaluate():
    """Evaluation returns valid metrics dict."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0, eval_mode=True, test_every=8)
    cfg = TrainingConfig(iterations=50, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    trainer.train(lambda s: None, callback_every=50)
    metrics = trainer.evaluate()

    assert "psnr" in metrics
    assert "ssim" in metrics
    assert "l1" in metrics
    assert "num_test" in metrics
    assert metrics["num_test"] > 0
    assert metrics["psnr"] > 10  # sanity — should be at least somewhat trained
    assert 0 < metrics["ssim"] < 1
    assert metrics["l1"] > 0


# ── Checkpoint tests ────────────────────────────────────────────────────────


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_save_load():
    """Save checkpoint, load it, verify state is preserved."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    splats_at_50 = trainer.splat_count

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)
        assert os.path.exists(ckpt_path)
        assert os.path.getsize(ckpt_path) > 1000

        # Load into a fresh trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        assert trainer2.iteration == 50
        assert trainer2.splat_count == splats_at_50
    finally:
        os.unlink(ckpt_path)


@pytest.mark.skipif(not HAS_GARDEN, reason="garden dataset not found")
def test_checkpoint_resume_training():
    """Train 50 → save → load → train 50 more. Verify it completes."""
    from msplat import TrainingConfig, Dataset, GaussianTrainer

    ds = Dataset(GARDEN, downscale_factor=4.0)
    cfg = TrainingConfig(iterations=100, num_downscales=0)
    trainer = GaussianTrainer(ds, cfg)

    for _ in range(50):
        trainer.step()

    with tempfile.NamedTemporaryFile(suffix=".msplat", delete=False) as f:
        ckpt_path = f.name

    try:
        trainer.save_checkpoint(ckpt_path)

        # Resume in a new trainer
        ds2 = Dataset(GARDEN, downscale_factor=4.0)
        cfg2 = TrainingConfig(iterations=100, num_downscales=0)
        trainer2 = GaussianTrainer(ds2, cfg2)
        trainer2.load_checkpoint(ckpt_path)

        for _ in range(50):
            stats = trainer2.step()

        assert trainer2.iteration == 100
        assert stats.splat_count > 0
        assert stats.ms_per_step > 0
    finally:
        os.unlink(ckpt_path)
