"""Tests for the similarity score (part 2)"""

import numpy as np

from rpca_dance.similarity import (
    biggest_drops, comparison_frame_range, dancer_box, find_offset, flip_left_right, iou_chance_level, iou_per_frame,
    normalize_silhouettes, sanity_checks,
)


def moving_square_video(number_of_frames: int) -> np.ndarray:
    """Toy masks: a 4 x 4 square that moves one pixel to the right every frame, on a 20 x 40 image"""
    masks = np.zeros((number_of_frames, 20, 40), dtype=bool)
    for frame in range(number_of_frames):
        masks[frame, 8:12, frame:frame + 4] = True
    return masks


def test_iou_per_frame():
    first = np.zeros((2, 1, 4), dtype=bool)
    first[:, 0, :2] = True
    second = first.copy()
    second[1, 0, :] = True  # frame 2: 2 shared pixels out of 4
    assert iou_per_frame(first, second).tolist() == [1.0, 0.5]


def test_box_and_normalization_remove_the_position():
    masks = np.zeros((1, 10, 10), dtype=bool)
    masks[0, 2:6, 3:7] = True
    assert dancer_box(masks, 0.0) == (2, 6, 3, 7)
    assert normalize_silhouettes(masks, dancer_box(masks, 0.0), 8, 8).all()  # the square fills the whole box


def test_flip_left_right():
    masks = np.array([[[True, False, False]]])
    assert flip_left_right(masks)[0, 0].tolist() == [False, False, True]


def test_find_offset_recovers_a_time_shift():
    reference = moving_square_video(30)
    learner = reference[5:]  # starts 5 frames later
    best_offset, _, _ = find_offset(reference, learner, start_offset=0, max_shift=8, min_overlap=10)
    assert best_offset == 5


def test_biggest_drops_are_apart():
    scores = np.ones(40)
    scores[10] = 0.0
    scores[11] = 0.1
    scores[30] = 0.2
    drops = biggest_drops(scores, np.arange(1, 41), count=2, window=4)
    drop_frames = [frame for frame, _ in drops]
    assert abs(drop_frames[0] - 11) <= 2  # around the two bad frames 11 and 12
    assert abs(drop_frames[1] - 31) <= 2


def test_sanity_checks_on_a_toy_video():
    reference = moving_square_video(30)
    checks = sanity_checks(reference, time_shift=3, max_shift=6, min_overlap=10, shift_pixels=1, erase_share=0.2, seed=0)
    assert checks["self"]["offset"] == 0 and checks["self"]["mean_score"] == 1.0
    assert checks["time shift"]["offset"] == 3
    assert checks["distorted"]["mean_score"] < 1.0


def test_comparison_frame_range_is_clipped():
    assert comparison_frame_range(400, 208, start_offset=0, max_shift=40) == (0, 248)
    assert comparison_frame_range(100, 80, start_offset=10, max_shift=40) == (0, 100)


def test_aligned_copy_beats_the_pixel_chance_level():
    reference = moving_square_video(30)
    level, best_values = iou_chance_level(reference, reference, 0, 3, 10, repeats=20, seed=0, percentile=95)
    assert len(best_values) == 20
    assert level < 1.0  # shuffled frames never match as well as the real order
