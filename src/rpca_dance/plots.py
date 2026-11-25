"""Figures, side-by-side montages, GIF/MP4 export"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_frame_image(frame: np.ndarray, output_path: Path, upscale_factor: int = 1) -> None:
    """Save one grayscale frame (values in [0, 1]) as a PNG file"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_values = np.clip(frame, 0.0, 1.0) * 255
    image = Image.fromarray(pixel_values.astype(np.uint8))
    new_size = (image.width * upscale_factor, image.height * upscale_factor) #upscale_factor makes the image bigger so it is easier to look at
    image = image.resize(new_size, Image.Resampling.NEAREST)
    image.save(output_path)


def save_contact_sheet(
    frames: np.ndarray, output_path: Path, number_of_rows: int = 4, number_of_columns: int = 5
) -> None:
    """Save a grid of evenly spaced frames as one PNG image

    Input: frames of shape (T, height, width), values in [0, 1]
    """
    number_of_frames, frame_height, frame_width = frames.shape
    number_of_tiles = number_of_rows * number_of_columns
    chosen_indices = np.linspace(0, number_of_frames - 1, number_of_tiles).astype(int)
    tiles = frames[chosen_indices].reshape(number_of_rows, number_of_columns, frame_height, frame_width)
    # Put the tiles of each row side by side, then stack the rows: (rows * height, columns * width)
    grid = tiles.transpose(0, 2, 1, 3).reshape(number_of_rows * frame_height, number_of_columns * frame_width)
    save_frame_image(grid, output_path, upscale_factor=2)


def plot_frame_differences(
    frame_differences: np.ndarray, frames_per_second: int, output_path: Path
) -> None:
    """Plot the frame-to-frame differences over time, with their median as a dashed line"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(1, len(frame_differences) + 1) / frames_per_second
    figure, axis = plt.subplots(figsize=(8, 3))
    axis.plot(times, frame_differences, color="black", linewidth=1)
    axis.axhline(np.median(frame_differences), color="gray", linestyle="--", label="median")
    axis.set_xlabel("time in the window (seconds)")
    axis.set_ylabel("mean |frame t+1 - frame t|")
    axis.set_ylim(bottom=0)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_history(
    history: dict, key: str, label: str, output_path: Path, log_scale: bool = False
) -> None:
    """Plot one quantity of a Robust PCA history (for example "errors" or "objective") against the iteration"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = history[key]
    iterations = np.arange(1, len(values) + 1)
    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(iterations, values, color="black", marker="o", markersize=3)
    if log_scale:
        axis.set_yscale("log")
    axis.set_xlabel("iteration")
    axis.set_ylabel(label)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
