"""Fixed-camera check"""

import numpy as np

from rpca_dance.video import camera_drift, check_fixed_camera


def test_a_cut_gives_a_big_jump():
    random_generator = np.random.default_rng(0)
    # 10 frames of the same scene with a little noise, then 10 frames of another scene
    first_scene = random_generator.random((6, 8))
    second_scene = random_generator.random((6, 8))
    scenes = np.stack([first_scene] * 10 + [second_scene] * 10)
    frames = scenes + 0.01 * random_generator.standard_normal(scenes.shape)
    check = check_fixed_camera(frames)
    assert check["largest_jump_frame"] == 10
    assert check["jump_ratio"] > 10


def test_camera_drift_finds_a_slow_pan():
    random_generator = np.random.default_rng(0)
    scene = random_generator.random((40, 60))
    # 4 blocks of 5 frames; the camera moves so the scene goes 2 pixels to the right in the last two blocks
    still = [scene] * 10
    moved = [np.roll(scene, 2, axis=1)] * 10
    shifts = camera_drift(np.stack(still + moved), 5)
    assert np.allclose(shifts[:2], 0, atol=0.1)
    assert np.allclose(shifts[2:, 0], 2, atol=0.1)
