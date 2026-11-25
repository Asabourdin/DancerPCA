"""F1 scores against hand-made masks, and run times"""

import itertools
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from rpca_dance.masks import resize_masks


def frame_numbers_to_annotate(number_of_frames: int, count: int) -> list[int]:
    """Pick `count` evenly spaced frame numbers between 1 and number_of_frames (1 = first frame)"""
    evenly_spaced = np.linspace(1, number_of_frames, count)
    return [int(round(value)) for value in evenly_spaced]


def export_frames_to_annotate(color_frames_dir: Path, output_dir: Path, frame_numbers: list[int]) -> None:
    """Copy the chosen color frames to output_dir, so the student can draw the masks on them

    The file names stay the same (frame_0001.png, ...), so each mask can be matched to its frame
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for frame_number in frame_numbers:
        file_name = f"frame_{frame_number:04d}.png"
        shutil.copy(color_frames_dir / file_name, output_dir / file_name)
    print(f"Exported {len(frame_numbers)} frames to annotate in {output_dir}")


def load_ground_truth(ground_truth_dir: Path, frame_height: int, frame_width: int) -> dict:
    """Load the hand-made masks frame_XXXX_mask.png (white dancers on black), at the given size

    Output: a dictionary {frame number: boolean mask of shape (frame_height, frame_width)}
    It is empty if no mask exists yet
    """
    truth = {}
    for mask_path in sorted(ground_truth_dir.glob("frame_*_mask.png")):
        frame_number = int(mask_path.name.split("_")[1])
        drawn_mask = np.asarray(Image.open(mask_path).convert("L")) > 127
        truth[frame_number] = resize_masks(drawn_mask[np.newaxis], frame_height, frame_width)[0]
    return truth


def f1_score(predicted_mask: np.ndarray, true_mask: np.ndarray) -> float:
    """F1 = 2 TP / (2 TP + FP + FN), where a positive is a dancer pixel

    TP: dancer pixels found, FP: background pixels wrongly marked as dancer,
    FN: dancer pixels missed
    -> 1 is perfect, 0 means no overlap at all
    """
    true_positives = np.sum(predicted_mask & true_mask)
    false_positives = np.sum(predicted_mask & ~true_mask)
    false_negatives = np.sum(~predicted_mask & true_mask)
    denominator = 2 * true_positives + false_positives + false_negatives
    if denominator == 0:
        return 1.0  # no dancer in the truth and none predicted
    return float(2 * true_positives / denominator)


def evaluate_methods(method_masks: dict, truth: dict) -> dict:
    """F1 score of every method on every annotated frame

    method_masks: {method name: boolean masks of shape (T, height, width)}, same size as the truth
    truth: {frame number: boolean mask}, from load_ground_truth
    Output: {method name: array of F1 scores, one per annotated frame, in frame order}
    """
    frame_numbers = sorted(truth)
    scores = {name: np.zeros(len(frame_numbers)) for name in method_masks}
    pairs = itertools.product(method_masks, enumerate(frame_numbers))
    for method_name, (position, frame_number) in pairs:
        predicted = method_masks[method_name][frame_number - 1]  # frame 1 is index 0
        scores[method_name][position] = f1_score(predicted, truth[frame_number])
    return scores


def evaluate_if_possible(method_masks: dict, ground_truth_dir: Path, frame_height: int, frame_width: int) -> dict | None:
    """Run evaluate_methods if hand-made masks exist; otherwise print a clear message and return None"""
    truth = load_ground_truth(ground_truth_dir, frame_height, frame_width)
    if len(truth) == 0:
        print(f"No hand-made masks found in {ground_truth_dir}: skipping the F1 evaluation.")
        return None
    print(f"Found {len(truth)} hand-made masks: computing F1 scores.")
    return evaluate_methods(method_masks, truth)


def describe_mask_file(mask_path: Path) -> dict:
    """Check one hand-made mask: its size, its image mode, and how many pixels are not pure black or white

    Gray pixels come from anti-aliasing on the edges of the drawing; transparent pixels come from an
    RGBA export. Both are handled by load_ground_truth (gray > 127 is a dancer, transparent pixels are black here)
    """
    image = Image.open(mask_path)
    gray_levels = np.asarray(image.convert("L"))
    in_between = (gray_levels > 0) & (gray_levels < 255)
    transparent_pixels = 0
    if image.mode == "RGBA":
        transparent_pixels = int(np.sum(np.asarray(image)[:, :, 3] == 0))
    return {
        "name": mask_path.name,
        "width": image.size[0],
        "height": image.size[1],
        "mode": image.mode,
        "gray_pixels": int(np.sum(in_between)),
        "transparent_pixels": transparent_pixels,
        "dancer_share": float(np.mean(gray_levels > 127)),
    }


def check_ground_truth_files(ground_truth_dir: Path, expected_width: int, expected_height: int) -> list[dict]:
    """Describe every hand-made mask and print one line per mask; warn if a size is wrong"""
    reports = []
    for mask_path in sorted(ground_truth_dir.glob("frame_*_mask.png")):
        report = describe_mask_file(mask_path)
        size_text = "ok"
        if report["width"] != expected_width or report["height"] != expected_height:
            size_text = "WRONG SIZE"
        print(
            f"{report['name']}: {report['width']} x {report['height']} ({size_text}), "
            f"mode {report['mode']}, {report['gray_pixels']} gray pixels, "
            f"{report['transparent_pixels']} transparent pixels, dancer share {report['dancer_share']:.4f}"
        )
        reports.append(report)
    return reports


def mask_iou(first_masks: np.ndarray, second_masks: np.ndarray) -> float:
    """Intersection over union of two sets of boolean masks of the same shape (1 = same masks)"""
    intersection = np.sum(first_masks & second_masks)
    union = np.sum(first_masks | second_masks)
    if union == 0:
        return 1.0
    return float(intersection / union)


def print_f1_table(scores: dict, frame_numbers: list[int], title: str) -> None:
    """Print the F1 scores: one row per method, one column per annotated frame, then the mean"""
    print(title)
    header = "".join([f"{number:>7d}" for number in frame_numbers])
    print(f"{'method':<24s}{header}{'mean':>8s}")
    for method_name, values in scores.items():
        row = "".join([f"{value:7.3f}" for value in values])
        print(f"{method_name:<24s}{row}{values.mean():8.3f}")


def precision_and_recall(method_masks: dict, truth: dict) -> dict:
    """Precision and recall of each method, pooled over all annotated frames

    precision = TP / (TP + FP): the share of marked pixels that are really dancer pixels (low with shadows)
    recall = TP / (TP + FN): the share of dancer pixels that are found (low with holes)
    Output: {method name: (precision, recall)}
    """
    frame_indices = [frame_number - 1 for frame_number in sorted(truth)]
    true_masks = np.stack([truth[frame_number] for frame_number in sorted(truth)])
    rates = {}
    for method_name, masks in method_masks.items():
        predicted = masks[frame_indices]
        true_positives = np.sum(predicted & true_masks)
        precision = true_positives / max(np.sum(predicted), 1)
        recall = true_positives / max(np.sum(true_masks), 1)
        rates[method_name] = (float(precision), float(recall))
    return rates
