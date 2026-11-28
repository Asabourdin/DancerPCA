"""The toy example of the notebook: a "video" of 3 frames with 4 pixels"""

import numpy as np


def make_toy_example() -> tuple[np.ndarray, np.ndarray]:
    """Build the toy video: an empty studio under changing light, plus one dancer

    Studio b = (0.5, 0.5, 0.2, 0.8), brightness 1.0 / 0.9 / 1.1 for the 3 frames
    The dancer adds +0.6 to pixel 3 in frame 2
    Output: the background (4 x 3, rank 1) and the dancer (4 x 3, one non-zero); M is their sum
    """
    studio = np.array([0.5, 0.5, 0.2, 0.8])
    brightness = np.array([1.0, 0.9, 1.1])
    # Each column is the studio times the brightness of that frame -> rank 1
    background = np.outer(studio, brightness)
    dancer = np.zeros((4, 3))
    dancer[2, 1] = 0.6  # pixel 3, frame 2 (Python counts from 0)
    return background, dancer


def singular_values(matrix: np.ndarray) -> np.ndarray:
    """Return the singular values of a matrix, from the largest to the smallest"""
    return np.linalg.svd(matrix, compute_uv=False)


def print_matrix(title: str, matrix: np.ndarray) -> None:
    """Print a small matrix with 3 decimals, under a title (rows = pixels, columns = frames)"""
    print(title)
    # Adding 0.0 turns -0.0 (from soft thresholding) into 0.0, so the zeros print without a minus sign
    print(np.array2string(matrix + 0.0, precision=3, suppress_small=True))
    print()
