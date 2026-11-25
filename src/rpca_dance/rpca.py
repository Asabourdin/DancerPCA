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


def estimate_noise_level(matrix: np.ndarray) -> float:
    """Estimate the noise standard deviation sigma with the MAD rule

    D = M minus the per-pixel median over time (the median background)
    sigma = 1.4826 * median(|D - median(D)|)
    The dancers are sparse, so the median mostly ignores them
    1.4826 makes this equal to sigma when the noise is Gaussian
    """
    deviations = matrix - np.median(matrix, axis=1, keepdims=True)
    absolute_deviations = np.abs(deviations - np.median(deviations))
    return float(1.4826 * np.median(absolute_deviations))


def noise_based_weights(noise_level: float, shape: tuple[int, int]) -> tuple[float, float]:
    """Choose lambda1 and lambda2 from the noise level, as in the Lasso chapter

    lambda must be at least the size of the noise in the dual norm of each penalty:
    - dual of the nuclear norm = largest singular value; for n x T noise it is about sigma (sqrt(n) + sqrt(T))
    - dual of the l1 norm = largest absolute entry; for n T Gaussian values it is about sigma sqrt(2 log(n T))
    Output: lambda1, lambda2
    """
    number_of_rows, number_of_columns = shape
    low_rank_weight = noise_level * (np.sqrt(number_of_rows) + np.sqrt(number_of_columns))
    sparse_weight = noise_level * np.sqrt(2 * np.log(number_of_rows * number_of_columns))
    return float(low_rank_weight), float(sparse_weight)


def candes_weights(low_rank_weight: float, shape: tuple[int, int]) -> tuple[float, float]:
    """Keep lambda1 and set lambda2 = lambda1 / sqrt(max(n, T)) (Candes, Li, Ma, Wright, 2011)

    Output: lambda1, lambda2
    """
    return low_rank_weight, float(low_rank_weight / np.sqrt(max(shape)))


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


def proximal_step(
    matrix: np.ndarray, low_rank_point: np.ndarray, sparse_point: np.ndarray,
    low_rank_weight: float, sparse_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One proximal gradient step, taken from the point (L, S)

    1. gradient step of size 1/2 on 1/2 ||M - L - S||^2: L + R/2 and S + R/2, with R = M - L - S
    2. proximal step: SVT with threshold lambda1/2 on L, soft thresholding with threshold lambda2/2 on S
    Output: the new L, the new S, and the singular values of the new L
    """
    residual = matrix - low_rank_point - sparse_point
    new_low_rank_part, singular_values = singular_value_threshold(low_rank_point + STEP_SIZE * residual, STEP_SIZE * low_rank_weight)
    new_sparse_part = soft_threshold(sparse_point + STEP_SIZE * residual, STEP_SIZE * sparse_weight)
    return new_low_rank_part, new_sparse_part, singular_values


def relative_change(new_part: np.ndarray, old_part: np.ndarray) -> float:
    """Return ||new - old||_F / ||new||_F (0 if new is all zeros)"""
    new_norm = np.linalg.norm(new_part)
    if new_norm == 0:
        return 0.0
    return float(np.linalg.norm(new_part - old_part) / new_norm)


def rpca_proximal_gradient(
    matrix: np.ndarray,
    low_rank_weight: float,
    sparse_weight: float,
    tolerance: float,
    max_iterations: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Robust PCA by proximal gradient descent (course, matrix completion lecture, chapter 4)

    Each iteration is one proximal_step from the current (L, S)
    With step size 1/2 the objective never goes up
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
        new_low_rank_part, sparse_part, singular_values = proximal_step(matrix, low_rank_part, sparse_part, low_rank_weight, sparse_weight)
        change = relative_change(new_low_rank_part, low_rank_part)
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


def rpca_fista(
    matrix: np.ndarray,
    low_rank_weight: float,
    sparse_weight: float,
    tolerance: float,
    max_iterations: int,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Robust PCA by accelerated proximal gradient descent (FISTA, course)

    Same proximal_step as rpca_proximal_gradient, but taken from an extrapolated point:
    Y = X_k + (t_k - 1) / t_(k+1) * (X_k - X_(k-1)), with t_(k+1) = (1 + sqrt(1 + 4 t_k^2)) / 2
    
    The objective does not always go down at every step, so we stop on the change instead: when ||L_new - L_old||_F / ||L_new||_F < tolerance, or after max_iterations
    Output: L, S and the history (objective, changes, rank, nonzeros, run_time)
    """
    start_time = time.perf_counter()
    low_rank_part = np.zeros_like(matrix)
    sparse_part = np.zeros_like(matrix)
    low_rank_point, sparse_point = low_rank_part, sparse_part
    momentum_counter = 1.0  # t_k in the formula
    history = new_history()
    for iteration in range(1, max_iterations + 1):
        new_low_rank_part, new_sparse_part, singular_values = proximal_step(matrix, low_rank_point, sparse_point, low_rank_weight, sparse_weight)
        next_momentum_counter = (1 + np.sqrt(1 + 4 * momentum_counter ** 2)) / 2
        momentum = (momentum_counter - 1) / next_momentum_counter
        low_rank_point = new_low_rank_part + momentum * (new_low_rank_part - low_rank_part)
        sparse_point = new_sparse_part + momentum * (new_sparse_part - sparse_part)
        change = relative_change(new_low_rank_part, low_rank_part)
        low_rank_part, sparse_part, momentum_counter = new_low_rank_part, new_sparse_part, next_momentum_counter
        objective = penalized_objective(matrix, low_rank_part, sparse_part, singular_values, low_rank_weight, sparse_weight)
        record_step(history, objective, change, singular_values, sparse_part)
        if verbose and iteration % 20 == 0:
            print(f"Running Robust PCA (FISTA): iteration {iteration}, objective {objective:.4e}, change {change:.1e}")
        if iteration > 1 and change < tolerance:
            break
    history["run_time"] = time.perf_counter() - start_time
    return low_rank_part, sparse_part, history
