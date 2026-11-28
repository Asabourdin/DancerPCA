"""Synthetic low-rank + sparse matrices and the success grid (phase transition diagram)"""

import itertools
from pathlib import Path

import numpy as np

from rpca_dance.baselines import classic_pca
from rpca_dance.rpca import rpca_fista


def make_low_rank_plus_sparse(
    size: int, rank: int, sparsity: float, random_generator: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Build a random test problem M = L0 + S0 of shape (size, size)

    L0 = dot prod of U and V.T with Gaussian U and V of shape (size, rank), scaled by 1/sqrt(size)
    (co its entries are about 1 in size whatever the size)
    S0: each entry is non-zero with probability sparsity, with value +1 or -1
    Output: L0 and S0 (M is their sum)
    """
    left_factor = random_generator.standard_normal((size, rank))
    right_factor = random_generator.standard_normal((size, rank))
    low_rank = left_factor @ right_factor.T / np.sqrt(size)
    is_corrupted = random_generator.random((size, size)) < sparsity
    random_signs = random_generator.choice([-1.0, 1.0], size=(size, size))
    sparse = is_corrupted * random_signs
    return low_rank, sparse


def relative_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """Return ||estimate - truth||_F / ||truth||_F"""
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def run_one_trial(
    size: int, rank: int, sparsity: float, low_rank_weight: float,
    tolerance: float, max_iterations: int, random_generator: np.random.Generator,
) -> dict:
    """Build one random problem, solve it with Robust PCA (FISTA) and with classic PCA

    lambda2 = lambda1 / sqrt(size)
    Output: a dictionary with the Frobenius errors ||L_hat - L0||_F of both methods,
    the rank found by Robust PCA, its number of iterations and its run time
    """
    low_rank, sparse = make_low_rank_plus_sparse(size, rank, sparsity, random_generator)
    matrix = low_rank + sparse
    sparse_weight = low_rank_weight / np.sqrt(size)
    rpca_low_rank, _, history = rpca_fista(matrix, low_rank_weight, sparse_weight, tolerance, max_iterations)
    pca_low_rank, _ = classic_pca(matrix, rank)
    return {
        "rpca_error": float(np.linalg.norm(rpca_low_rank - low_rank)),
        "rpca_rank": history["rank"][-1],
        "pca_error": float(np.linalg.norm(pca_low_rank - low_rank)),
        "iterations": len(history["objective"]),
        "run_time": history["run_time"],
    }


def run_simulation_grid(
    size: int, rank_ratios: list[float], sparsities: list[float], number_of_trials: int,
    low_rank_weight: float, tolerance: float, max_iterations: int, seed: int,
) -> dict:
    """Run every (rank ratio, sparsity, trial) combination once

    The true rank is round(rank_ratio * size)
    Output: a dictionary of arrays of shape (number of ranks, number of sparsities, number of trials)
    for "rpca_error", "rpca_rank", "pca_error", "iterations", "run_time",
    plus the grid itself ("true_ranks", "rank_ratios", "sparsities")
    """
    random_generator = np.random.default_rng(seed)
    true_ranks = np.round(np.array(rank_ratios) * size).astype(int)
    shape = (len(rank_ratios), len(sparsities), number_of_trials)
    results = {key: np.zeros(shape) for key in ["rpca_error", "rpca_rank", "pca_error", "iterations", "run_time"]}
    # All the (rank index, sparsity index, trial) triples, so that a single loop is enough
    all_cells = list(itertools.product(range(len(rank_ratios)), range(len(sparsities)), range(number_of_trials)))
    for rank_index, sparsity_index, trial in all_cells:
        trial_result = run_one_trial(
            size, int(true_ranks[rank_index]), sparsities[sparsity_index],
            low_rank_weight, tolerance, max_iterations, random_generator,
        )
        store_trial(results, trial_result, (rank_index, sparsity_index, trial))
        if trial == number_of_trials - 1:
            print(f"Simulation: rank {true_ranks[rank_index]}, sparsity {sparsities[sparsity_index]:.2f} done")
    results["true_ranks"] = true_ranks
    results["rank_ratios"] = np.array(rank_ratios)
    results["sparsities"] = np.array(sparsities)
    return results


def store_trial(results: dict, trial_result: dict, position: tuple[int, int, int]) -> None:
    """Copy the numbers of one trial into the result arrays, at the given (rank, sparsity, trial) position"""
    for key in trial_result:
        results[key][position] = trial_result[key]


def success_fraction(results: dict, low_rank_weight: float, factor: float) -> tuple[np.ndarray, np.ndarray]:
    """Share of successful trials in each cell, for Robust PCA and for classic PCA

    Robust PCA succeeds if ||L_hat - L0||_F < factor * lambda1 * sqrt(r) AND the rank of L_hat is r
    (lambda1 * sqrt(r) is the shrinkage bias: SVT lowers each of the r singular values by about lambda1)
    Classic PCA is given the true rank, so only the error condition applies to it
    Output: two arrays of shape (number of ranks, number of sparsities), values in [0, 1]
    """
    # Reshape the true ranks to (number of ranks, 1, 1) so they line up with the result arrays
    true_ranks = results["true_ranks"].reshape(-1, 1, 1)
    threshold = factor * low_rank_weight * np.sqrt(true_ranks)
    rpca_success = (results["rpca_error"] < threshold) & (results["rpca_rank"] == true_ranks)
    pca_success = results["pca_error"] < threshold
    return rpca_success.mean(axis=2), pca_success.mean(axis=2)


def save_results(results: dict, output_path: Path) -> None:
    """Save the simulation results in a .npz file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **results)


def load_results(input_path: Path) -> dict:
    """Load simulation results saved by save_results"""
    saved = np.load(input_path)
    return {key: saved[key] for key in saved.files}


def summarize_success(results: dict, low_rank_weight: float, factor: float) -> dict:
    """Count the successes of the grid, for one success factor

    Output: a dictionary with the number of trials, the successful Robust PCA trials, the cells where all trials
    succeed, the successful classic PCA trials, the median and smallest classic PCA error divided by the bias
    lambda1 * sqrt(r), and the total and largest Robust PCA run times (seconds)
    """
    rpca_map, pca_map = success_fraction(results, low_rank_weight, factor)
    number_of_trials = results["rpca_error"].shape[2]
    shrinkage_bias = low_rank_weight * np.sqrt(results["true_ranks"].reshape(-1, 1, 1))
    pca_error_ratios = results["pca_error"] / shrinkage_bias
    return {
        "trials": results["rpca_error"].size,
        "rpca_successful_trials": int(round(rpca_map.sum() * number_of_trials)),
        "rpca_full_cells": int((rpca_map == 1).sum()),
        "cells": rpca_map.size,
        "pca_successful_trials": int(round(pca_map.sum() * number_of_trials)),
        "pca_median_error_ratio": float(np.median(pca_error_ratios)),
        "pca_smallest_error_ratio": float(pca_error_ratios.min()),
        "total_run_time": float(results["run_time"].sum()),
        "largest_run_time": float(results["run_time"].max()),
    }
