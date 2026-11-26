"""Analysis of the video: masks of every method, F1 scores, resolution comparison and the lighting stress test

Robust PCA runs are loaded from results/decompositions/ when they exist (see decomposition.py)
"""

import itertools
import time
from pathlib import Path

import numpy as np

from rpca_dance.baselines import classic_pca, temporal_median
from rpca_dance.decomposition import decomposition_path, run_decomposition, run_with_weights
from rpca_dance.lighting import apply_brightness, brightness_ramp, brightness_step
from rpca_dance.masks import foreground_masks
from rpca_dance.video import frames_to_matrix


def print_run_summary(setting_name: str, result: dict, masks: np.ndarray) -> None:
    """One line with the lambdas, the rank of L, the number of iterations, the run time and the mask share"""
    print(
        f"{setting_name}: lambda1 {result['low_rank_weight']:.4f}, lambda2 {result['sparse_weight']:.5f}, "
        f"sigma {result['noise_level']:.5f}, rank of L {result['rank']}, {result['iterations']} iterations, "
        f"{result['run_time']:.0f} s, dancer share {masks.mean():.4f}"
    )


def rpca_masks_by_setting(
    matrix: np.ndarray, settings: list[tuple[str, float]], decompositions_dir: Path, run_name: str,
    tolerance: float, max_iterations: int, frame_height: int, frame_width: int, threshold: float,
) -> tuple[dict, dict]:
    """Run (or load) Robust PCA for every (rule, multiplier) setting and turn each S into masks

    Output: {"RPCA candes x1.0": masks of shape (T, height, width)} and {same name: run result}
    """
    masks = {}
    results = {}
    for rule, multiplier in settings:
        output_path = decomposition_path(decompositions_dir, run_name, rule, multiplier)
        result = run_decomposition(matrix, rule, multiplier, tolerance, max_iterations, output_path)
        name = f"RPCA {rule} x{multiplier}"
        masks[name] = foreground_masks(result["sparse"], frame_height, frame_width, threshold)
        results[name] = result
        print_run_summary(name, result, masks[name])
    return masks, results


def all_settings(rules: list[str], multipliers: list[float]) -> list[tuple[str, float]]:
    """Every (rule, multiplier) pair, in a fixed order"""
    return list(itertools.product(rules, multipliers))


def baseline_masks(
    matrix: np.ndarray, frame_height: int, frame_width: int, threshold: float, pca_rank: int
) -> tuple[dict, dict]:
    """Masks and run times of the two quick baselines, classic PCA and the temporal median

    Both use the same threshold on |S| as Robust PCA, so only the background model changes
    """
    start_time = time.perf_counter()
    _, pca_foreground = classic_pca(matrix, pca_rank)
    pca_time = time.perf_counter() - start_time
    start_time = time.perf_counter()
    _, median_foreground = temporal_median(matrix)
    median_time = time.perf_counter() - start_time
    masks = {
        "classic PCA": foreground_masks(pca_foreground, frame_height, frame_width, threshold),
        "temporal median": foreground_masks(median_foreground, frame_height, frame_width, threshold),
    }
    return masks, {"classic PCA": pca_time, "temporal median": median_time}


def lighting_gains(number_of_frames: int, ramp_amplitude: float, step_frames: list[int], step_gain: float) -> dict:
    """Gains of every brightness change: one slow ramp, and one sudden step per step frame

    Output: {"ramp": gains, "step at 100": gains, ...}, each an array of T gains
    """
    gains = {"ramp": brightness_ramp(number_of_frames, ramp_amplitude)}
    for step_frame in step_frames:
        gains[f"step at {step_frame}"] = brightness_step(number_of_frames, step_frame, step_gain)
    return gains


def lighting_run_path(decompositions_dir: Path, change_name: str) -> Path:
    """File of the Robust PCA run for one brightness change, for example 160x90_lighting_step_at_100.npz"""
    return decompositions_dir / f"160x90_lighting_{change_name.replace(' ', '_')}.npz"


def lighting_test_masks(
    lit_matrix: np.ndarray, reference_result: dict, output_path: Path, tolerance: float, max_iterations: int,
    frame_height: int, frame_width: int, threshold: float, pca_rank: int,
) -> tuple[dict, dict, dict]:
    """Robust PCA, classic PCA and the temporal median on a matrix with a brightness change

    Robust PCA keeps the lambdas of the original video (reference_result): the dancers and the
    noise do not change, only the lighting does
    Output: masks by method, run times by method, and the Robust PCA run
    """
    result = run_with_weights(
        lit_matrix, reference_result["low_rank_weight"], reference_result["sparse_weight"],
        reference_result["noise_level"], tolerance, max_iterations, output_path,
    )
    masks = {"Robust PCA": foreground_masks(result["sparse"], frame_height, frame_width, threshold)}
    times = {"Robust PCA": result["run_time"]}
    quick_masks, quick_times = baseline_masks(lit_matrix, frame_height, frame_width, threshold, pca_rank)
    masks.update(quick_masks)
    times.update(quick_times)
    return masks, times, result


def run_lighting_test(
    matrix: np.ndarray, gains: dict, reference_result: dict, decompositions_dir: Path, tolerance: float,
    max_iterations: int, frame_height: int, frame_width: int, threshold: float, pca_rank: int,
) -> dict:
    """Apply each brightness change to M, then get the masks of the three methods (one loop over the changes)

    Output: {change name: masks by method}
    """
    masks_by_change = {}
    for change_name, change_gains in gains.items():
        print(f"Lighting test: {change_name}")
        lit_matrix = apply_brightness(matrix, change_gains)
        output_path = lighting_run_path(decompositions_dir, change_name)
        masks, _, result = lighting_test_masks(
            lit_matrix, reference_result, output_path, tolerance, max_iterations,
            frame_height, frame_width, threshold, pca_rank,
        )
        print_run_summary(f"Robust PCA, {change_name}", result, masks["Robust PCA"])
        masks_by_change[change_name] = masks
    return masks_by_change


def video_masks(
    frames: np.ndarray, decompositions_dir: Path, run_name: str, rule: str, tolerance: float,
    max_iterations: int, threshold: float,
) -> tuple[np.ndarray, dict]:
    """Robust PCA (multiplier 1) on a video, then the dancer masks |S| > threshold

    Input: frames of shape (T, height, width)
    Output: boolean masks of shape (T, height, width), and the run result
    """
    _, frame_height, frame_width = frames.shape
    matrix = frames_to_matrix(frames)
    output_path = decomposition_path(decompositions_dir, run_name, rule, 1.0)
    result = run_decomposition(matrix, rule, 1.0, tolerance, max_iterations, output_path)
    masks = foreground_masks(result["sparse"], frame_height, frame_width, threshold)
    print_run_summary(run_name, result, masks)
    return masks, result
