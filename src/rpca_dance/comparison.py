"""Part 2 pipeline: from the reference masks and a learner video to the pixel score and the rhythm score

Any learner video can be plugged in: a YouTube link or a local file (main.py --learner)
"""

from pathlib import Path

import numpy as np

from rpca_dance.experiments import video_masks
from rpca_dance.rhythm import (
    autocorrelation, passage_coverage, passage_selection, rhythm_chance_level, rhythm_score, selected_chance_level,
    selected_correlation, smoothed_motion_curve, sparse_motion_curve,
)
from rpca_dance.similarity import (
    comparison_frame_range, dancer_box, flip_left_right, iou_chance_level, moving_average, normalize_silhouettes,
    similarity_score,
)
from rpca_dance.video import matrix_to_frames, prepare_video, video_name


def prepare_learner(
    source: str, frames_root: Path, raw_video_dir: Path, start_seconds: float, duration_seconds: float,
    frames_per_second: int, frame_width: int, frame_height: int, decompositions_dir: Path,
    rule: str, tolerance: float, max_iterations: int, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a learner video exactly like the reference, run Robust PCA on it and build its masks

    Output: the masks (T, height, width) and S as frames (T, height, width)
    """
    name = video_name(source)
    frames = prepare_video(
        source, frames_root / f"learner_{name}", raw_video_dir, start_seconds, duration_seconds,
        frames_per_second, frame_width, frame_height,
    )
    run_name = f"learner_{name}_{frame_width}x{frame_height}"
    masks, result = video_masks(frames, decompositions_dir, run_name, rule, tolerance, max_iterations, threshold)
    return masks, matrix_to_frames(result["sparse"], frame_height, frame_width)


def comparison_boxes(
    reference_masks: np.ndarray, learner_masks: np.ndarray, start_offset: int, max_shift: int, outlier_share: float
) -> tuple[tuple, tuple]:
    """Dancer boxes: for the reference, only on the frames that the offset search can pair with the learner"""
    first, last = comparison_frame_range(len(reference_masks), len(learner_masks), start_offset, max_shift)
    reference_box = dancer_box(reference_masks[first:last], outlier_share)
    learner_box = dancer_box(learner_masks, outlier_share)
    return reference_box, learner_box


def comparison_silhouettes(
    reference_masks: np.ndarray, learner_masks: np.ndarray, start_offset: int, max_shift: int,
    outlier_share: float, width: int, height: int,
) -> tuple[np.ndarray, np.ndarray, tuple, tuple]:
    """Crop both videos to their dancer box and resize them to width x height

    Output: reference silhouettes, learner silhouettes, reference box, learner box
    """
    reference_box, learner_box = comparison_boxes(reference_masks, learner_masks, start_offset, max_shift, outlier_share)
    reference_silhouettes = normalize_silhouettes(reference_masks, reference_box, width, height)
    learner_silhouettes = normalize_silhouettes(learner_masks, learner_box, width, height)
    return reference_silhouettes, learner_silhouettes, reference_box, learner_box


def search_settings(start_offset: int, max_shift: int, manual_offset: int | None) -> tuple[int, int]:
    """With a manual offset, the search is reduced to that single offset (shift 0)"""
    if manual_offset is None:
        return start_offset, max_shift
    return manual_offset, 0


def pixel_comparison(
    reference_silhouettes: np.ndarray, learner_silhouettes: np.ndarray, start_offset: int, max_shift: int,
    min_overlap: int, repeats: int, seed: int, percentile: float,
) -> dict:
    """Pixel score as filmed and mirrored, the better one, and its chance level (shuffled frames, same search)

    Output: {"as filmed": result, "mirrored": result, "kept": name of the better one, "chance level": value}
    """
    as_filmed = similarity_score(reference_silhouettes, learner_silhouettes, start_offset, max_shift, min_overlap, None)
    mirrored_learner = flip_left_right(learner_silhouettes)
    mirrored = similarity_score(reference_silhouettes, mirrored_learner, start_offset, max_shift, min_overlap, None)
    kept = "as filmed"
    kept_learner = learner_silhouettes
    if mirrored["mean_score"] > as_filmed["mean_score"]:
        kept = "mirrored"
        kept_learner = mirrored_learner
    chance, _ = iou_chance_level(reference_silhouettes, kept_learner, start_offset, max_shift, min_overlap, repeats, seed, percentile)
    return {"as filmed": as_filmed, "mirrored": mirrored, "kept": kept, "chance level": chance}


def rhythm_comparison(
    reference_curve: np.ndarray, learner_curve: np.ndarray, start_offset: int, max_shift: int, min_overlap: int,
    min_circular_shift: int, percentile: float, local_window: int, number_of_moments: int,
) -> dict:
    """Rhythm score of two (smoothed) motion curves, with its chance level (circular shifts, same search)

    Output: the result of rhythm_score, plus "chance level" and "chance values"
    """
    result = rhythm_score(reference_curve, learner_curve, start_offset, max_shift, min_overlap, local_window, number_of_moments)
    chance, chance_values = rhythm_chance_level(
        reference_curve, learner_curve, start_offset, max_shift, min_overlap, min_circular_shift, percentile,
    )
    result["chance level"] = chance
    result["chance values"] = chance_values
    return result


def motion_curves(
    reference_silhouettes: np.ndarray, learner_silhouettes: np.ndarray, smoothing_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Main version: smoothed motion curves of the two sets of silhouettes"""
    reference_curve = smoothed_motion_curve(reference_silhouettes, smoothing_window)
    learner_curve = smoothed_motion_curve(learner_silhouettes, smoothing_window)
    return reference_curve, learner_curve


def sparse_motion_curves(
    reference_sparse: np.ndarray, learner_sparse: np.ndarray, reference_box: tuple, learner_box: tuple, smoothing_window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Robustness check only: smoothed motion curves computed from S instead of the masks"""
    reference_curve = moving_average(sparse_motion_curve(reference_sparse, reference_box), smoothing_window)
    learner_curve = moving_average(sparse_motion_curve(learner_sparse, learner_box), smoothing_window)
    return reference_curve, learner_curve


def repetition_check(reference_curve: np.ndarray, min_lag: int, max_lag: int) -> tuple[int, float]:
    """Strongest repetition of the reference movement for lags from min_lag to max_lag

    Output: the lag (in frames) with the highest autocorrelation, and that autocorrelation
    """
    lags, correlations = autocorrelation(reference_curve, min_lag, max_lag)
    best = int(np.argmax(correlations))
    return int(lags[best]), float(correlations[best])


def known_offset_rhythm(
    reference_curve: np.ndarray, learner_curve: np.ndarray, known_offset: int, min_overlap: int,
    min_circular_shift: int, percentile: float, local_window: int, number_of_moments: int,
) -> dict:
    """Rhythm score at a known offset, with no search: the real use case, where the app plays the music

    It is rhythm_comparison with a search reduced to one offset. The chance level then takes no maximum:
    it is the percentile of the correlations at the circular shifts
    """
    return rhythm_comparison(
        reference_curve, learner_curve, known_offset, 0, min_overlap, min_circular_shift, percentile,
        local_window, number_of_moments,
    )


def passage_rhythm(
    reference_curve: np.ndarray, learner_curve: np.ndarray, known_offset: int, passages_seconds: list,
    window_start_seconds: float, frames_per_second: int, min_circular_shift: int, percentile: float,
) -> dict:
    """Rhythm score at the known offset on the frames of some passages only (for example the unison passages)

    The chance level uses the circular shifts of the whole learner curve, evaluated on the same frames
    Output: {"score", "chance level", "chance values", "coverage": (covered values, values) for each passage}
    """
    number_of_frames = len(reference_curve) + 1
    selection = passage_selection(number_of_frames, passages_seconds, window_start_seconds, frames_per_second)
    score = selected_correlation(reference_curve, learner_curve, known_offset, selection)
    chance, chance_values = selected_chance_level(
        reference_curve, learner_curve, known_offset, selection, min_circular_shift, percentile,
    )
    single_selections = []
    for passage in passages_seconds:
        single_selections.append(passage_selection(number_of_frames, [passage], window_start_seconds, frames_per_second))
    coverage = passage_coverage(reference_curve, learner_curve, known_offset, single_selections)
    return {"score": score, "chance level": chance, "chance values": chance_values, "coverage": coverage}
