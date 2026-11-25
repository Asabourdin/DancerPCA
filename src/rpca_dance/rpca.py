"""Robust PCA solved by proximal gradient descent on the penalized problem

Problem: minimize F(L, S) = 1/2 ||M - L - S||_F^2 + lambda1 ||L||_* + lambda2 ||S||_1
The algorithm only uses the two proximal operators of operators.py
"""

import time

import numpy as np

from rpca_dance.operators import singular_value_threshold, soft_threshold

# The gradient of 1/2 ||M - L - S||^2 with respect to the pair (L, S) is
# (-(M - L - S), -(M - L - S)). It is 2-Lipschitz, so the step size is 1/2.
STEP_SIZE = 0.5


def default_lambda(matrix: np.ndarray) -> float:
    """Return 1 / sqrt(max(n, T)), the usual ratio lambda2 / lambda1 (Candes, Li, Ma, Wright, 2011)"""
    return 1.0 / np.sqrt(max(matrix.shape))


def new_history() -> dict:
    """Return an empty history: one list per quantity we follow, plus the run time"""
    return {"objective": [], "changes": [], "rank": [], "nonzeros": [], "run_time": 0.0}


def record_step(history: dict, objective: float, change: float, singular_values: np.ndarray, sparse_part: np.ndarray) -> None:
    """Add the numbers of one iteration to the history"""
    history["objective"].append(objective)
    history["changes"].append(change)
    history["rank"].append(int(np.count_nonzero(singular_values)))
    history["nonzeros"].append(int(np.count_nonzero(sparse_part)))


def penalized_objective(
    matrix: np.ndarray, low_rank_part: np.ndarray, sparse_part: np.ndarray,
    singular_values: np.ndarray, low_rank_weight: float, sparse_weight: float,
) -> float:
    """Value of F(L, S) = 1/2 ||M - L - S||_F^2 + lambda1 ||L||_* + lambda2 ||S||_1

    singular_values are the singular values of L (their sum is ||L||_*)
    """
    fit_term = 0.5 * np.linalg.norm(matrix - low_rank_part - sparse_part) ** 2
    low_rank_term = low_rank_weight * singular_values.sum()
    sparse_term = sparse_weight * np.abs(sparse_part).sum()
    return fit_term + low_rank_term + sparse_term


def rpca_proximal_gradient(
    matrix: np.ndarray,
    low_rank_weight: float,
    sparse_weight: float,
    tolerance: float,
    max_iterations: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Robust PCA by proximal gradient descent (course, matrix completion lecture, chapter 4)

    Each iteration:
      1. gradient step of size 1/2 on 1/2 ||M - L - S||^2: L + R/2 and S + R/2, with R = M - L - S
      2. proximal step: SVT with threshold lambda1/2 on L, soft thresholding with threshold lambda2/2 on S
    With this step size the objective never goes up
    Stops when the objective drops by less than `tolerance` (relative), or after max_iterations
    "changes" in the history is ||L_new - L_old||_F / ||L_new||_F
    Output: L, S and the history (objective, changes, rank, nonzeros, run_time)
    """
    start_time = time.perf_counter()
    low_rank_part = np.zeros_like(matrix)
    sparse_part = np.zeros_like(matrix)
    history = new_history()
    previous_objective = np.inf
    for iteration in range(1, max_iterations + 1):
        residual = matrix - low_rank_part - sparse_part
        new_low_rank_part, singular_values = singular_value_threshold(low_rank_part + STEP_SIZE * residual, STEP_SIZE * low_rank_weight)
        sparse_part = soft_threshold(sparse_part + STEP_SIZE * residual, STEP_SIZE * sparse_weight)
        change = np.linalg.norm(new_low_rank_part - low_rank_part) / max(np.linalg.norm(new_low_rank_part), 1e-12)
        low_rank_part = new_low_rank_part
        objective = penalized_objective(matrix, low_rank_part, sparse_part, singular_values, low_rank_weight, sparse_weight)
        record_step(history, objective, change, singular_values, sparse_part)
        if verbose and iteration % 20 == 0:
            print(f"Running Robust PCA: iteration {iteration}, objective {objective:.4e}, change {change:.1e}")
        if previous_objective - objective < tolerance * abs(previous_objective):
            break
        previous_objective = objective
    history["run_time"] = time.perf_counter() - start_time
    return low_rank_part, sparse_part, history
