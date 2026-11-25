"""Tests for the simulation grid and the success rule"""

import numpy as np

from rpca_dance.simulation import make_low_rank_plus_sparse, success_fraction


def test_generated_matrices_have_the_right_rank_and_sparsity():
    random_generator = np.random.default_rng(0)
    low_rank, sparse = make_low_rank_plus_sparse(100, 7, 0.1, random_generator)
    assert np.linalg.matrix_rank(low_rank) == 7
    assert set(np.unique(sparse)) <= {-1.0, 0.0, 1.0}
    assert 0.08 < np.mean(sparse != 0) < 0.12


def test_success_needs_a_small_error_and_the_right_rank():
    # One rank (r = 4), one sparsity, three trials; lambda1 = 0.1 and factor 3 give a threshold of 0.6
    results = {
        "true_ranks": np.array([4]),
        "rpca_error": np.array([[[0.5, 0.5, 0.9]]]),
        "rpca_rank": np.array([[[4, 5, 4]]]),
        "pca_error": np.array([[[0.1, 2.0, 2.0]]]),
    }
    rpca_share, pca_share = success_fraction(results, low_rank_weight=0.1, factor=3)
    # Trial 1 succeeds; trial 2 has the wrong rank; trial 3 has a too large error
    assert np.isclose(rpca_share[0, 0], 1 / 3)
    assert np.isclose(pca_share[0, 0], 1 / 3)
