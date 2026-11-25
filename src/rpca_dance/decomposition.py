"""Run Robust PCA on a video matrix for a given lambda rule, and cache the result on disk"""

import itertools
from pathlib import Path

import numpy as np

from rpca_dance.rpca import candes_weights, estimate_noise_level, noise_based_weights, rpca_fista


def choose_weights(matrix: np.ndarray, rule: str, multiplier: float) -> tuple[float, float, float]:
    """Pick lambda1 and lambda2 for the video, then multiply both by `multiplier`

    rule "noise": lambda1 and lambda2 from the noise level (Lasso chapter)
    rule "candes": same lambda1, and lambda2 = lambda1 / sqrt(max(n, T)) (Candes et al., 2011)
    Output: lambda1, lambda2, and the estimated noise level sigma
    """
    noise_level = estimate_noise_level(matrix)
    low_rank_weight, sparse_weight = noise_based_weights(noise_level, matrix.shape)
    if rule == "candes":
        low_rank_weight, sparse_weight = candes_weights(low_rank_weight, matrix.shape)
    elif rule != "noise":
        raise ValueError(f"Unknown lambda rule: {rule}")
    return multiplier * low_rank_weight, multiplier * sparse_weight, noise_level


def decomposition_path(output_dir: Path, run_name: str, rule: str, multiplier: float) -> Path:
    """File name of one cached run, for example results/decompositions/160x90_candes_x1.0.npz"""
    return output_dir / f"{run_name}_{rule}_x{multiplier}.npz"


def run_decomposition(
    matrix: np.ndarray, rule: str, multiplier: float, tolerance: float,
    max_iterations: int, output_path: Path,
) -> dict:
    """Pick lambda1 and lambda2 with `rule` and `multiplier`, then run (or load) Robust PCA

    Output: the dictionary of run_with_weights
    """
    if output_path.exists():
        print(f"Loading saved decomposition: {output_path.name}")
        return load_decomposition(output_path)
    low_rank_weight, sparse_weight, noise_level = choose_weights(matrix, rule, multiplier)
    print(f"Running Robust PCA ({rule} rule, x{multiplier})")
    return run_with_weights(matrix, low_rank_weight, sparse_weight, noise_level, tolerance, max_iterations, output_path)


def run_with_weights(
    matrix: np.ndarray, low_rank_weight: float, sparse_weight: float, noise_level: float,
    tolerance: float, max_iterations: int, output_path: Path,
) -> dict:
    """Run Robust PCA (FISTA) once with given lambdas and save L, S and the main numbers; load them if already saved

    noise_level is only stored, to know later where the lambdas came from
    L and S are saved as float32 to halve the file size
    Output: a dictionary with "low_rank", "sparse", "low_rank_weight", "sparse_weight",
    "noise_level", "run_time", "iterations", "rank", "objective"
    """
    if output_path.exists():
        print(f"Loading saved decomposition: {output_path.name}")
        return load_decomposition(output_path)
    print(f"Running Robust PCA: lambda1 = {low_rank_weight:.4f}, lambda2 = {sparse_weight:.5f}")
    low_rank, sparse, history = rpca_fista(matrix, low_rank_weight, sparse_weight, tolerance, max_iterations, verbose=True)
    result = {
        "low_rank": low_rank.astype(np.float32),
        "sparse": sparse.astype(np.float32),
        "low_rank_weight": low_rank_weight,
        "sparse_weight": sparse_weight,
        "noise_level": noise_level,
        "run_time": history["run_time"],
        "iterations": len(history["objective"]),
        "rank": history["rank"][-1],
        "objective": np.array(history["objective"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)
    print(f"Done in {history['run_time']:.0f} s, {result['iterations']} iterations, rank of L = {result['rank']}")
    return result


def load_decomposition(input_path: Path) -> dict:
    """Load a run saved by run_decomposition. Numbers come back as plain floats and ints"""
    saved = np.load(input_path)
    result = {key: saved[key] for key in saved.files}
    for key in ["low_rank_weight", "sparse_weight", "noise_level", "run_time"]:
        result[key] = float(result[key])
    result["iterations"] = int(result["iterations"])
    result["rank"] = int(result["rank"])
    return result


def run_lambda_comparison(
    matrix: np.ndarray, rules: list[str], multipliers: list[float], tolerance: float,
    max_iterations: int, output_dir: Path, run_name: str,
) -> dict:
    """Run (or load) every combination of lambda rule and multiplier

    Output: a dictionary {(rule, multiplier): result of run_decomposition}
    """
    results = {}
    for rule, multiplier in itertools.product(rules, multipliers):
        output_path = decomposition_path(output_dir, run_name, rule, multiplier)
        results[(rule, multiplier)] = run_decomposition(matrix, rule, multiplier, tolerance, max_iterations, output_path)
    return results
