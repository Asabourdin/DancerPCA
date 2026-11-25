"""Fixed-camera check"""

import numpy as np

from rpca_dance.video import check_fixed_camera


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
