"""Tests for the motion rhythm score"""

import numpy as np

from rpca_dance.rhythm import (
    autocorrelation, curve_correlation, find_rhythm_offset, local_correlation, motion_curve, passage_selection,
    rhythm_chance_level, selected_chance_level, selected_pairs,
)


def test_motion_curve_counts_changed_pixels():
    silhouettes = np.zeros((3, 2, 2), dtype=bool)
    silhouettes[1, 0, 0] = True  # one pixel turns on, then off again
    silhouettes[2, 1, :] = True  # two more pixels turn on
    assert motion_curve(silhouettes).tolist() == [0.25, 0.75]


def test_correlation_ignores_scale_and_level():
    curve = np.array([0.0, 1.0, 0.0, 2.0])
    assert np.isclose(curve_correlation(curve, 10 * curve + 3), 1.0)
    assert np.isclose(curve_correlation(curve, -curve), -1.0)


def test_rhythm_offset_is_found():
    random_generator = np.random.default_rng(0)
    reference = random_generator.random(100)
    learner = reference[7:]  # starts 7 values later
    best_offset, _, correlations = find_rhythm_offset(reference, learner, 0, 10, 50)
    assert best_offset == 7
    assert np.isclose(correlations.max(), 1.0)


def test_chance_level_uses_the_same_search():
    random_generator = np.random.default_rng(1)
    reference = random_generator.random(120)
    level, best_values = rhythm_chance_level(reference, reference, 0, 5, 60, min_circular_shift=16, percentile=95)
    assert len(best_values) == 120 - 2 * 16 + 1
    assert level < 0.99  # a random curve shifted in a circle does not match itself


def test_local_correlation_and_autocorrelation():
    wave = np.sin(np.arange(64) * 2 * np.pi / 16)  # repeats every 16 values
    assert np.allclose(local_correlation(wave, wave, 8), 1.0)
    lags, correlations = autocorrelation(wave, 4, 20)
    assert lags[np.argmax(correlations)] == 16


def test_passage_selection_keeps_the_right_frames():
    # 2 frames per second, window starting at 10 s: frame k is at 10 + (k - 1) / 2 seconds
    selection = passage_selection(8, [(11.0, 12.0), (13.0, 20.0)], 10.0, 2)
    assert np.flatnonzero(selection).tolist() == [2, 3, 6, 7]  # frames 3, 4, 7 and 8


def test_selected_pairs_use_reference_frame_numbers():
    reference = np.arange(10.0)  # value t belongs to frame t + 2 (frames counted from 1)
    learner = np.arange(10.0)
    selection = np.zeros(11, dtype=bool)
    selection[[4, 5]] = True  # frames 5 and 6
    reference_part, _, frame_numbers = selected_pairs(reference, learner, 2, selection)
    assert frame_numbers.tolist() == [5, 6]
    assert reference_part.tolist() == [3.0, 4.0]


def test_known_offset_chance_level_is_the_search_reduced_to_one_offset():
    random_generator = np.random.default_rng(0)
    reference = random_generator.random(120)
    learner = random_generator.random(100)
    keep_all = np.ones(121, dtype=bool)
    without_search, _ = selected_chance_level(reference, learner, 5, keep_all, 16, 95)
    search_of_one_offset, _ = rhythm_chance_level(reference, learner, 5, 0, 80, 16, 95)
    assert np.isclose(without_search, search_of_one_offset)
