"""Synthetic low-rank + sparse matrices and the success grid"""

import numpy as np


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
