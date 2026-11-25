"""Tests for the lighting stress test and for the median as an l1 fit"""

import numpy as np

from rpca_dance.baselines import temporal_median
from rpca_dance.lighting import apply_brightness, brightness_ramp, brightness_step


def test_ramp_goes_from_darker_to_brighter():
    gains = brightness_ramp(5, 0.2)
    assert np.allclose(gains, [0.8, 0.9, 1.0, 1.1, 1.2])


def test_step_changes_the_gain_after_the_step_frame():
    gains = brightness_step(4, step_frame=2, step_gain=0.8)
    assert gains.tolist() == [1.0, 1.0, 0.8, 0.8]


def test_a_gain_keeps_a_constant_background_at_rank_one_and_clips():
    studio = np.array([[0.2], [0.5], [0.9]])
    matrix = np.repeat(studio, 4, axis=1)
    lit = apply_brightness(matrix, brightness_ramp(4, 0.2))
    assert np.linalg.matrix_rank(lit[:2]) == 1  # the two pixels that are not clipped
    assert lit.max() == 1.0  # 0.9 * 1.2 is clipped to 1


def test_median_is_the_best_constant_in_l1():
    # The median c minimizes sum_t |m_t - c|: no other constant does better
    pixel_over_time = np.array([[0.5, 0.5, 0.5, 0.5, 1.0]])  # a dancer passes once
    background, _ = temporal_median(pixel_over_time)
    candidates = np.linspace(0, 1, 101)
    l1_losses = np.abs(pixel_over_time.T - candidates).sum(axis=0)
    assert np.isclose(background[0, 0], 0.5)
    assert np.isclose(np.abs(pixel_over_time - background).sum(), l1_losses.min())
