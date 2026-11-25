"""Tests for the frames <-> matrix conversion."""

import numpy as np

from rpca_dance.video import frames_to_matrix, is_url, matrix_to_frames


def test_round_trip_gives_back_the_same_frames():
    random_generator = np.random.default_rng(0)
    frames = random_generator.random((5, 3, 4))  # 5 frames of 3 x 4 pixels
    matrix = frames_to_matrix(frames)
    assert matrix.shape == (12, 5)
    frames_back = matrix_to_frames(matrix, frame_height=3, frame_width=4)
    assert np.array_equal(frames, frames_back)


def test_each_column_is_one_frame():
    frames = np.arange(2 * 2 * 3).reshape(2, 2, 3)  # 2 frames of 2 x 3 pixels
    matrix = frames_to_matrix(frames)
    assert np.array_equal(matrix[:, 1], frames[1].ravel())


def test_is_url():
    assert is_url("https://www.youtube.com/watch?v=gSm-eTRl9JE")
    assert not is_url("data/raw/learner.mp4")
