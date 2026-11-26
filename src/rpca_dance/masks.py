"""Turn S (or any foreground) into binary masks, change their size, crop and resize silhouettes"""

import cv2
import numpy as np

from rpca_dance.video import matrix_to_frames


def foreground_masks(
    foreground_matrix: np.ndarray, frame_height: int, frame_width: int, threshold: float
) -> np.ndarray:
    """A pixel is a dancer pixel when |foreground| > threshold

    Input: a matrix of shape (height * width, T), for example S
    Output: a boolean array of shape (T, height, width)
    """
    foreground_frames = matrix_to_frames(foreground_matrix, frame_height, frame_width)
    return np.abs(foreground_frames) > threshold


def otsu_threshold(foreground_matrix: np.ndarray) -> float:
    """Pick the threshold on |foreground| automatically with Otsu's method

    Otsu chooses the gray level that best splits the values into two groups (the two groups have the smallest total variance). Values are clipped to [0, 1]
    Output: a threshold between 0 and 1
    """
    gray_levels = (np.clip(np.abs(foreground_matrix), 0, 1) * 255).astype(np.uint8)
    otsu_level, _ = cv2.threshold(gray_levels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(otsu_level) / 255


def resize_masks(masks: np.ndarray, frame_height: int, frame_width: int) -> np.ndarray:
    """Change the size of binary masks of shape (T, height, width)

    Each new pixel takes the share of dancer pixels it covers (area averaging), and is a dancer pixel if this share is above one half
    """
    resized = []
    for mask in masks:
        shares = cv2.resize(mask.astype(np.float32), (frame_width, frame_height), interpolation=cv2.INTER_AREA)
        resized.append(shares > 0.5)
    return np.stack(resized)


def keep_region(frames: np.ndarray, region: tuple[int, int, int, int] | None) -> np.ndarray:
    """Set everything outside a fixed region of interest to zero (False for masks)

    Used for a learner filmed outdoors: people walking far behind the dancers end up in S, but outside the region
    frames: masks or S, of shape (T, height, width); region: (top, bottom, left, right), bottom and right excluded,
    or None to keep the whole frame
    """
    if region is None:
        return frames
    top, bottom, left, right = region
    inside = np.zeros(frames.shape[1:], dtype=bool)
    inside[top:bottom, left:right] = True
    return frames * inside
