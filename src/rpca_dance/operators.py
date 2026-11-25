"""Proximal operators: soft thresholding and singular value thresholding"""

import numpy as np


def soft_threshold(matrix: np.ndarray, threshold: float) -> np.ndarray:
    """Shrink every entry towards 0 by threshold, and set small entries to exactly 0

    Formula, entry by entry: sign(x) * max(|x| - threshold, 0)
    -> the operator of the l1 norm: it gives the sparse part
    """
    shrunk_sizes = np.maximum(np.abs(matrix) - threshold, 0.0)
    return np.sign(matrix) * shrunk_sizes


def singular_value_threshold(matrix: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Soft-threshold the singular values of a matrix, then rebuild it

    It is the proximal operator of the nuclear norm: it gives the low-rank part
    Singular values below threshold become 0, so the rank goes down
    Output: the rebuilt matrix, and its new singular values (their number of non-zeros is the rank, their sum is the nuclear norm)
    """
    # Reduced SVD: matrix = dot prof of U, diag(singular_values), Vt
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(matrix, full_matrices=False)
    new_singular_values = soft_threshold(singular_values, threshold)
    # Multiplying the columns of U by the new values is the same as the dot prod of U and diag(values), but faster
    rebuilt_matrix = (left_vectors * new_singular_values) @ right_vectors_transposed
    return rebuilt_matrix, new_singular_values
