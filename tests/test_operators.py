"""Tests for soft thresholding and singular value thresholding"""

import numpy as np

from rpca_dance.operators import singular_value_threshold, soft_threshold


def test_soft_threshold_on_small_numbers():
    values = np.array([3.0, -0.4, -2.5, 0.0])
    expected = np.array([2.0, 0.0, -1.5, 0.0])
    assert np.allclose(soft_threshold(values, 1.0), expected)


def test_singular_value_threshold_shrinks_the_singular_values():
    # A diagonal matrix has its diagonal as singular values: 5, 2 and 0.5
    matrix = np.diag([5.0, 2.0, 0.5])
    rebuilt_matrix, new_singular_values = singular_value_threshold(matrix, 1.0)
    assert np.allclose(new_singular_values, [4.0, 1.0, 0.0])
    assert np.allclose(rebuilt_matrix, np.diag([4.0, 1.0, 0.0]))


def test_singular_value_threshold_lowers_the_rank():
    random_generator = np.random.default_rng(0)
    rank_two = random_generator.standard_normal((20, 2)) @ random_generator.standard_normal((2, 10))
    small_noise = 0.01 * random_generator.standard_normal((20, 10))
    _, new_singular_values = singular_value_threshold(rank_two + small_noise, 1.0)
    assert np.count_nonzero(new_singular_values) == 2
