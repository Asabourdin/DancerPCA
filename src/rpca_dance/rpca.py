"""Robust PCA: constrained version (inexact ALM) and penalized version

Both versions only use the two operators of operators.py
"""

import time

import numpy as np

from rpca_dance.operators import singular_value_threshold, soft_threshold


def default_lambda(matrix: np.ndarray) -> float:
    """Return the standard weight of the sparse part: 1 / sqrt(max(n, T))"""
    return 1.0 / np.sqrt(max(matrix.shape))


def new_history() -> dict:
    """Return an empty history: one list per quantity we follow, plus the run time"""
    return {"errors": [], "objective": [], "rank": [], "nonzeros": [], "run_time": 0.0}


def record_step(history: dict, error: float, objective: float, singular_values: np.ndarray, sparse_part: np.ndarray) -> None:
    """Add the numbers of one iteration to the history"""
    history["errors"].append(error)
    history["objective"].append(objective)
    history["rank"].append(int(np.count_nonzero(singular_values)))
    history["nonzeros"].append(int(np.count_nonzero(sparse_part)))


def initial_dual_variable(matrix: np.ndarray, sparse_weight: float) -> np.ndarray:
    """Starting value of the Lagrange multiplier Y

    Y = M / J(M), with J(M) = max(spectral norm of M, max |M_ij| / lambda)
    """
    spectral_norm = np.linalg.norm(matrix, ord=2)
    scaled_max_entry = np.abs(matrix).max() / sparse_weight
    return matrix / max(spectral_norm, scaled_max_entry)


def rpca_constrained(
    matrix: np.ndarray,
    tolerance: float,
    max_iterations: int,
    mu_growth: float,
    sparse_weight: float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Principal Component Pursuit: min ||L||_* + lambda ||S||_1 subject to L + S = M

    Solved with the inexact augmented Lagrange multiplier method 
    Each iteration: L = SVT(M - S + Y/mu, 1/mu), S = soft(M - L + Y/mu, lambda/mu), then Y moves by mu * (M - L - S), and mu grows by mu_growth
    -> stops when ||M - L - S||_F / ||M||_F < tolerance, or after max_iterations
    Output: L, S and the history (errors, objective, rank, nonzeros, run_time)
    """
    start_time = time.perf_counter()
    if sparse_weight is None:
        sparse_weight = default_lambda(matrix)
    matrix_norm = np.linalg.norm(matrix)
    dual_variable = initial_dual_variable(matrix, sparse_weight)
    mu = 1.25 / np.linalg.norm(matrix, ord=2)  # usual starting value from the paper
    mu_max = mu * 1e7  # mu stops growing here, so the steps stay stable
    sparse_part = np.zeros_like(matrix)
    history = new_history()
    for iteration in range(1, max_iterations + 1):
        low_rank_part, singular_values = singular_value_threshold(matrix - sparse_part + dual_variable / mu, 1.0 / mu)
        sparse_part = soft_threshold(matrix - low_rank_part + dual_variable / mu, sparse_weight / mu)
        residual = matrix - low_rank_part - sparse_part
        dual_variable = dual_variable + mu * residual
        mu = min(mu * mu_growth, mu_max)
        error = np.linalg.norm(residual) / matrix_norm
        objective = singular_values.sum() + sparse_weight * np.abs(sparse_part).sum()
        record_step(history, error, objective, singular_values, sparse_part)
        if verbose and iteration % 10 == 0:
            print(f"Running Robust PCA: iteration {iteration}, error {error:.1e}")
        if error < tolerance:
            break
    history["run_time"] = time.perf_counter() - start_time
    return low_rank_part, sparse_part, history


def penalized_objective(
    matrix: np.ndarray, low_rank_part: np.ndarray, sparse_part: np.ndarray,
    singular_values: np.ndarray, low_rank_weight: float, sparse_weight: float,
) -> float:
    """Value of 1/2 ||M - L - S||_F^2 + lambda1 ||L||_* + lambda2 ||S||_1

    singular_values are the singular values of L (their sum is ||L||_*)
    """
    fit_term = 0.5 * np.linalg.norm(matrix - low_rank_part - sparse_part) ** 2
    low_rank_term = low_rank_weight * singular_values.sum()
    sparse_term = sparse_weight * np.abs(sparse_part).sum()
    return fit_term + low_rank_term + sparse_term


def rpca_penalized(
    matrix: np.ndarray,
    low_rank_weight: float,
    sparse_weight: float,
    tolerance: float,
    max_iterations: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Minimize 1/2 ||M - L - S||_F^2 + lambda1 ||L||_* + lambda2 ||S||_1

    This is the statistical model with noise: M = L + S + small noise
    Alternating exact minimization: L = SVT(M - S, lambda1), then S = soft(M - L, lambda2)
    Each step can only lower the objective, so the objective never goes up
    -> Stops when the objective drops by less than tolerance (relative), or after max_iterations
    In the history, "errors" is the relative change of L between two iterations
    Output: L, S and the history (errors, objective, rank, nonzeros, run_time)
    """
    start_time = time.perf_counter()
    low_rank_part = np.zeros_like(matrix)
    sparse_part = np.zeros_like(matrix)
    history = new_history()
    previous_objective = np.inf
    for iteration in range(1, max_iterations + 1):
        previous_low_rank_part = low_rank_part
        low_rank_part, singular_values = singular_value_threshold(matrix - sparse_part, low_rank_weight)
        sparse_part = soft_threshold(matrix - low_rank_part, sparse_weight)
        objective = penalized_objective(matrix, low_rank_part, sparse_part, singular_values, low_rank_weight, sparse_weight)
        change = np.linalg.norm(low_rank_part - previous_low_rank_part) / max(np.linalg.norm(low_rank_part), 1e-12)
        record_step(history, change, objective, singular_values, sparse_part)
        if verbose and iteration % 10 == 0:
            print(f"Running penalized Robust PCA: iteration {iteration}, objective {objective:.4e}")
        if previous_objective - objective < tolerance * abs(previous_objective):
            break
        previous_objective = objective
    history["run_time"] = time.perf_counter() - start_time
    return low_rank_part, sparse_part, history
