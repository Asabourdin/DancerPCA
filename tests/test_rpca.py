"""Tests for both versions of Robust PCA, on the toy example and on a random matrix"""

import numpy as np

import config
from rpca_dance.baselines import classic_pca
from rpca_dance.rpca import rpca_constrained, rpca_penalized
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


def test_constrained_rpca_finds_the_dancer_in_the_toy_example():
    background, dancer = make_toy_example()
    low_rank_part, sparse_part, history = rpca_constrained(
        background + dancer, config.RPCA_TOLERANCE, config.RPCA_MAX_ITERATIONS, config.ALM_MU_GROWTH
    )
    assert np.allclose(low_rank_part, background, atol=1e-5)
    assert np.allclose(sparse_part, dancer, atol=1e-5)
    assert history["rank"][-1] == 1


def test_penalized_rpca_puts_the_dancer_in_s_only():
    background, dancer = make_toy_example()
    _, sparse_part, history = rpca_penalized(
        background + dancer, config.TOY_LOW_RANK_WEIGHT, config.TOY_SPARSE_WEIGHT,
        config.PENALIZED_TOLERANCE, config.PENALIZED_MAX_ITERATIONS,
    )
    # Only pixel 3 in frame 2 is non-zero: its value is shrunk a bit (the penalty adds bias)
    assert np.count_nonzero(sparse_part) == 1
    assert sparse_part[2, 1] > 0.5


def test_constrained_rpca_recovers_a_random_matrix_and_the_error_goes_down():
    random_generator = np.random.default_rng(config.RANDOM_SEED)
    low_rank, sparse = make_low_rank_plus_sparse(100, 5, 0.05, random_generator)
    low_rank_part, _, history = rpca_constrained(
        low_rank + sparse, config.RPCA_TOLERANCE, config.RPCA_MAX_ITERATIONS, config.ALM_MU_GROWTH
    )
    relative_error = np.linalg.norm(low_rank_part - low_rank) / np.linalg.norm(low_rank)
    assert relative_error < config.SIMULATION_SUCCESS_TOLERANCE
    assert history["errors"][-1] < config.RPCA_TOLERANCE
    assert history["errors"][-1] < history["errors"][0]


def test_penalized_objective_never_goes_up():
    random_generator = np.random.default_rng(config.RANDOM_SEED)
    low_rank, sparse = make_low_rank_plus_sparse(100, 5, 0.05, random_generator)
    noise = 0.01 * random_generator.standard_normal((100, 100))
    _, _, history = rpca_penalized(
        low_rank + sparse + noise, 0.5, 0.05, config.PENALIZED_TOLERANCE, config.PENALIZED_MAX_ITERATIONS
    )
    objective_steps = np.diff(history["objective"])
    assert np.all(objective_steps <= 1e-10)
