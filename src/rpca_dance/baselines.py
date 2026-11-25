"""Baselines for Robust PCA: classic PCA (l2 fit, low rank) and the temporal median (l1 fit, constant background)"""

import numpy as np


def classic_pca(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Classic PCA as a background model: keep the best rank-r approximation of M

    Truncated SVD: L = sum of the first `rank` singular triplets, S = M - L
    It is the best fit in the least-squares sense, so a large outlier pulls L towards it ("blends" into the background)
    Output: L and S
    """
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(matrix, full_matrices=False)
    low_rank_part = (left_vectors[:, :rank] * singular_values[:rank]) @ right_vectors_transposed[:rank]
    return low_rank_part, matrix - low_rank_part


def temporal_median(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Background = the median of each pixel over time; foreground = M - background

    Each dancer covers a pixel less than half of the time, so the median gives the empty studio
    Output: L (the median column repeated T times) and S = M - L
    """
    median_background = np.median(matrix, axis=1, keepdims=True)
    low_rank_part = np.repeat(median_background, matrix.shape[1], axis=1)
    return low_rank_part, matrix - low_rank_part
