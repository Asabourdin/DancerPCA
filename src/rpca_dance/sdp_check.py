"""Small CVXPY check of the convex problem

CVXPY rewrites the nuclear norm as a semidefinite program (SDP) and gives it to a
general-purpose conic solver. We use it as a reference answer for our own solver
"""

import time

import cvxpy as cp
import numpy as np

from rpca_dance.rpca import rpca_fista
from rpca_dance.simulation import make_low_rank_plus_sparse


def solve_with_cvxpy(
    matrix: np.ndarray, low_rank_weight: float, sparse_weight: float
) -> tuple[np.ndarray, np.ndarray, float, float, str]:
    """Solve min 1/2 ||M - L - S||_F^2 + lambda1 ||L||_* + lambda2 ||S||_1 with CVXPY

    cp.normNuc is the nuclear norm, cp.norm1 the sum of the absolute values of all entries
    Output: L, S, the optimal objective value, the run time in seconds, the solver name
    """
    low_rank_variable = cp.Variable(matrix.shape)
    sparse_variable = cp.Variable(matrix.shape)
    fit_term = 0.5 * cp.sum_squares(matrix - low_rank_variable - sparse_variable)
    penalty = low_rank_weight * cp.normNuc(low_rank_variable) + sparse_weight * cp.norm1(sparse_variable)
    problem = cp.Problem(cp.Minimize(fit_term + penalty))
    start_time = time.perf_counter()
    problem.solve()
    run_time = time.perf_counter() - start_time
    solver_name = problem.solver_stats.solver_name
    return low_rank_variable.value, sparse_variable.value, float(problem.value), run_time, solver_name


def make_check_problem(size: int, rank: int, sparsity: float, seed: int) -> tuple[np.ndarray, float, float]:
    """Build a random M = L0 + S0 for the check, with lambda1 = 0.1 and lambda2 = lambda1 / sqrt(size)

    Output: M, lambda1, lambda2
    """
    random_generator = np.random.default_rng(seed)
    low_rank, sparse = make_low_rank_plus_sparse(size, rank, sparsity, random_generator)
    low_rank_weight = 0.1
    return low_rank + sparse, low_rank_weight, low_rank_weight / np.sqrt(size)


def compare_with_cvxpy(
    size: int, rank: int, sparsity: float, seed: int, tolerance: float, max_iterations: int
) -> dict:
    """Solve the same problem with our FISTA solver and with CVXPY, and compare the answers

    Output: a dictionary with both objective values, the relative gap between them, the relative differences between the two L and the two S, both run times, the solver name
    """
    matrix, low_rank_weight, sparse_weight = make_check_problem(size, rank, sparsity, seed)
    our_low_rank, our_sparse, history = rpca_fista(matrix, low_rank_weight, sparse_weight, tolerance, max_iterations)
    cvx_low_rank, cvx_sparse, cvx_objective, cvx_time, solver_name = solve_with_cvxpy(matrix, low_rank_weight, sparse_weight)
    our_objective = history["objective"][-1]
    return {
        "our_objective": our_objective,
        "cvxpy_objective": cvx_objective,
        "objective_gap": abs(our_objective - cvx_objective) / cvx_objective,
        "low_rank_difference": np.linalg.norm(our_low_rank - cvx_low_rank) / np.linalg.norm(cvx_low_rank),
        "sparse_difference": np.linalg.norm(our_sparse - cvx_sparse) / np.linalg.norm(cvx_sparse),
        "our_time": history["run_time"],
        "cvxpy_time": cvx_time,
        "solver": solver_name,
    }


def time_both_solvers(
    sizes: list[int], rank: int, sparsity: float, seed: int, tolerance: float, max_iterations: int
) -> dict:
    """Run compare_with_cvxpy for each size, and collect the run times and objective gaps

    Output: a dictionary of arrays, one value per size: "sizes", "our_time", "cvxpy_time", "objective_gap"
    """
    timings = {"sizes": np.array(sizes), "our_time": [], "cvxpy_time": [], "objective_gap": []}
    for size in sizes:
        print(f"SDP check: size {size} x {size}...")
        comparison = compare_with_cvxpy(size, rank, sparsity, seed, tolerance, max_iterations)
        timings["our_time"].append(comparison["our_time"])
        timings["cvxpy_time"].append(comparison["cvxpy_time"])
        timings["objective_gap"].append(comparison["objective_gap"])
    timings["our_time"] = np.array(timings["our_time"])
    timings["cvxpy_time"] = np.array(timings["cvxpy_time"])
    timings["objective_gap"] = np.array(timings["objective_gap"])
    return timings
