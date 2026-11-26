"""Tests for the motion rhythm score"""

import numpy as np

from rpca_dance.rhythm import (
    autocorrelation, curve_correlation, find_rhythm_offset, local_correlation, motion_curve, rhythm_chance_level,
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
