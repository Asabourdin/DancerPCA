"""Lighting stress test: add an artificial brightness change to the video matrix M"""

import numpy as np


def brightness_ramp(number_of_frames: int, amplitude: float) -> np.ndarray:
    """Gain of each frame for a slow ramp, from 1 - amplitude (first frame) to 1 + amplitude (last frame)

    Example: amplitude 0.2 makes the first frame 20 % darker and the last one 20 % brighter
    """
    return np.linspace(1 - amplitude, 1 + amplitude, number_of_frames)


def brightness_step(number_of_frames: int, step_frame: int, step_gain: float) -> np.ndarray:
    """Gain of each frame for a sudden step, like a phone's auto-exposure

    Frames 1 to step_frame keep their brightness (gain 1), the next ones are multiplied by step_gain
    """
    gains = np.ones(number_of_frames)
    gains[step_frame:] = step_gain
    return gains


def apply_brightness(matrix: np.ndarray, gains: np.ndarray) -> np.ndarray:
    """Multiply each column (frame) of M by its gain, then clip to [0, 1] like a camera would

    A gain g_t on frame t turns the studio b into g_t * b: the background b g^T still has rank 1,
    but it is no longer constant over time
    """
    brighter_matrix = matrix * gains[np.newaxis, :]
    return np.clip(brighter_matrix, 0, 1)
