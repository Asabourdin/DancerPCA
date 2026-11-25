"""Tests for Robust PCA (proximal gradient descent), on the toy example and on a random matrix"""

import numpy as np

import config
from rpca_dance.baselines import classic_pca
from rpca_dance.rpca import rpca_fista, rpca_proximal_gradient
from rpca_dance.simulation import make_low_rank_plus_sparse


def make_toy_example() -> tuple[np.ndarray, np.ndarray]:
    """Toy video to check our code before acc running on the real vid: 4 pixels, 3 frames, one dancer on pixel 3 in frame 2"""
    studio = np.array([0.5, 0.5, 0.2, 0.8])
    brightness = np.array([1.0, 0.9, 1.1])
    background = np.outer(studio, brightness)
    dancer = np.zeros((4, 3))
    dancer[2, 1] = 0.6
    return background, dancer


def test_classic_pca_smears_the_dancer():
    background, dancer = make_toy_example()
    _, sparse_part = classic_pca(background + dancer, rank=1)
    # The dancer is not isolated: other pixels get non-zero values in S
    assert np.count_nonzero(np.abs(sparse_part) > 0.01) > 1


def test_robust_pca_puts_the_dancer_in_s_only():
    background, dancer = make_toy_example()
    _, sparse_part, history = rpca_proximal_gradient(
        background + dancer, config.TOY_LOW_RANK_WEIGHT, config.TOY_SPARSE_WEIGHT,
        config.RPCA_TOLERANCE, config.RPCA_MAX_ITERATIONS,
    )
    # Only pixel 3 in frame 2 is non-zero: its value is shrunk a bit (the penalty adds bias)
    assert np.count_nonzero(sparse_part) == 1
    assert sparse_part[2, 1] > 0.5
    assert history["rank"][-1] == 1


def test_robust_pca_recovers_a_random_matrix_up_to_the_shrinkage_bias():
    random_generator = np.random.default_rng(config.RANDOM_SEED)
    rank = 5
    low_rank, sparse = make_low_rank_plus_sparse(100, rank, 0.05, random_generator)
    low_rank_weight = 0.1
    low_rank_part, _, history = rpca_proximal_gradient(
        low_rank + sparse, low_rank_weight, low_rank_weight / 10, config.RPCA_TOLERANCE, config.RPCA_MAX_ITERATIONS
    )
    # SVT shrinks each of the r singular values by about lambda1, so ||L_hat - L0||_F is about lambda1 * sqrt(r)
    shrinkage_bias = low_rank_weight * np.sqrt(rank)
    assert np.linalg.norm(low_rank_part - low_rank) < 2 * shrinkage_bias
    assert history["rank"][-1] == rank


def test_objective_never_goes_up():
    random_generator = np.random.default_rng(config.RANDOM_SEED)
    low_rank, sparse = make_low_rank_plus_sparse(100, 5, 0.05, random_generator)
    noise = 0.01 * random_generator.standard_normal((100, 100))
    _, _, history = rpca_proximal_gradient(
        low_rank + sparse + noise, 0.5, 0.05, config.RPCA_TOLERANCE, config.RPCA_MAX_ITERATIONS
    )
    objective_steps = np.diff(history["objective"])
    assert np.all(objective_steps <= 1e-10)


def test_fista_reaches_the_same_objective_as_proximal_gradient():
    random_generator = np.random.default_rng(config.RANDOM_SEED)
    low_rank, sparse = make_low_rank_plus_sparse(100, 5, 0.05, random_generator)
    matrix = low_rank + sparse
    _, _, slow_history = rpca_proximal_gradient(matrix, 0.1, 0.01, 1e-12, 20000)
    _, _, fast_history = rpca_fista(matrix, 0.1, 0.01, config.FISTA_TOLERANCE, config.RPCA_MAX_ITERATIONS)
    # FISTA may go up for a step, so we only check the end point and the speed
    assert abs(fast_history["objective"][-1] - slow_history["objective"][-1]) < 1e-5 * slow_history["objective"][-1]
    assert fast_history["objective"][-1] < fast_history["objective"][0]
    assert len(fast_history["objective"]) < len(slow_history["objective"])
