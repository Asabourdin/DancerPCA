"""Figures, side-by-side montages, GIF/MP4 export"""

import itertools
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


def plot_success_maps(
    success_maps: list[np.ndarray], titles: list[str],
    rank_ratios: np.ndarray, sparsities: np.ndarray, output_path: Path,
) -> None:
    """Draw one phase transition diagram per map, side by side

    Each map has shape (number of rank ratios, number of sparsities), values in [0, 1]
    White = every trial succeeded, black = every trial failed
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    number_of_maps = len(success_maps)
    figure, axes = plt.subplots(1, number_of_maps, figsize=(5 * number_of_maps, 4.5), squeeze=False, layout="constrained")
    for axis, success_map, title in zip(axes[0], success_maps, titles):
        image = axis.imshow(success_map, origin="lower", cmap="gray", vmin=0, vmax=1)
        axis.set_xticks(range(len(sparsities)), [f"{value:.2f}" for value in sparsities], rotation=90)
        axis.set_yticks(range(len(rank_ratios)), [f"{value:.2f}" for value in rank_ratios])
        axis.set_xlabel("sparsity rho (share of corrupted entries)")
        axis.set_ylabel("rank ratio r / n")
        axis.set_title(title)
    figure.colorbar(image, ax=axes[0].tolist(), label="share of successful trials")
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_solver_times(timings: dict, output_path: Path) -> None:
    """Plot the run time of our solver and of CVXPY against the matrix size (log scale)"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 3.5))
    axis.plot(timings["sizes"], timings["our_time"], color="black", marker="o", label="our FISTA solver")
    axis.plot(timings["sizes"], timings["cvxpy_time"], color="gray", marker="s", linestyle="--", label="CVXPY (SDP)")
    axis.set_yscale("log")
    axis.set_xlabel("matrix size n (n x n)")
    axis.set_ylabel("run time (seconds)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def build_montage_frames(
    original: np.ndarray, background: np.ndarray, dancers: np.ndarray, masks: np.ndarray, dancer_gain: float
) -> np.ndarray:
    """Put four videos side by side: original | background L | dancers |S| | mask

    All inputs have shape (T, height, width). |S| is multiplied by dancer_gain so it is easier to see
    Output: an array of shape (T, height, 4 * width) with values in [0, 1]
    """
    visible_dancers = np.clip(np.abs(dancers) * dancer_gain, 0, 1)
    panels = [np.clip(original, 0, 1), np.clip(background, 0, 1), visible_dancers, masks.astype(float)]
    return np.concatenate(panels, axis=2)


def save_montage_image(montage_frames: np.ndarray, frame_indices: list[int], output_path: Path) -> None:
    """Stack a few montage frames (one per row) and save them as one PNG, 2 times larger"""
    stacked_rows = np.concatenate(montage_frames[frame_indices], axis=0)
    save_frame_image(stacked_rows, output_path, upscale_factor=2)


def save_montage_gif(montage_frames: np.ndarray, frames_per_second: int, output_path: Path) -> None:
    """Save the montage frames as an animated GIF, at the speed of the sampled video"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = []
    for frame in montage_frames:
        pixel_values = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        images.append(Image.fromarray(pixel_values))
    duration_ms = int(1000 / frames_per_second)
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)


def plot_mask_grid(
    mask_rows: list[np.ndarray], row_titles: list[str], frame_indices: list[int], output_path: Path
) -> None:
    """Grid of masks: one row per method or setting, one column per chosen frame

    mask_rows: a list of boolean arrays of shape (T, height, width); sizes may differ between rows
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    number_of_rows = len(mask_rows)
    number_of_columns = len(frame_indices)
    figure, axes = plt.subplots(number_of_rows, number_of_columns, figsize=(2.4 * number_of_columns, 1.5 * number_of_rows), squeeze=False)
    cells = itertools.product(range(number_of_rows), range(number_of_columns))
    for row, column in cells:
        axis = axes[row, column]
        axis.imshow(mask_rows[row][frame_indices[column]], cmap="gray", vmin=0, vmax=1)
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            axis.set_ylabel(row_titles[row], rotation=0, ha="right", va="center", fontsize=8)
        if row == 0:
            axis.set_title(f"frame {frame_indices[column] + 1}", fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_f1_scores(scores: dict, title: str, output_path: Path) -> None:
    """F1 score of each method: one gray dot per annotated frame and a black bar for the mean

    scores: {method name: array of F1 scores, one per annotated frame}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_names = list(scores)
    figure, axis = plt.subplots(figsize=(1.1 * len(method_names) + 2, 3.5))
    for position, method_name in enumerate(method_names):
        values = scores[method_name]
        axis.bar(position, values.mean(), width=0.6, color="lightgray", edgecolor="black")
        axis.scatter(np.full(len(values), position), values, color="black", s=10, zorder=3)
    axis.set_xticks(range(len(method_names)))
    axis.set_xticklabels(method_names, rotation=30, ha="right", fontsize=8)
    axis.set_ylim(0, 1)
    axis.set_ylabel("F1 score")
    axis.set_title(title, fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_brightness_gains(gains: dict, output_path: Path) -> None:
    """Gain of each frame for the lighting stress test (1 = original brightness)

    gains: {name of the change: array of T gains}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 2.8))
    line_styles = ["-", "--", ":"]
    for position, change_name in enumerate(gains):
        frame_numbers = np.arange(1, len(gains[change_name]) + 1)
        axis.plot(frame_numbers, gains[change_name], color="black", linestyle=line_styles[position], label=change_name)
    axis.set_xlabel("frame")
    axis.set_ylabel("brightness gain")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
