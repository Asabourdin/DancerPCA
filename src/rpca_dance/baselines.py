"""Baselines: classic PCA, temporal median, MOG2, pretrained segmentation"""

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
