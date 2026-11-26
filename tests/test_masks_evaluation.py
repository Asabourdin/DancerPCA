"""Tests for the masks and the F1 evaluation"""

import numpy as np
from PIL import Image

from rpca_dance.baselines import temporal_median
from rpca_dance.evaluation import (
    describe_mask_file, evaluate_if_possible, f1_score, frame_numbers_to_annotate, load_ground_truth, mask_iou,
)
from rpca_dance.masks import foreground_masks, keep_region, resize_masks


def test_f1_score_on_a_small_example():
    truth = np.array([[True, True, False, False]])
    predicted = np.array([[True, False, True, False]])
    # TP = 1, FP = 1, FN = 1, so F1 = 2 / (2 + 1 + 1) = 0.5
    assert f1_score(predicted, truth) == 0.5
    assert f1_score(truth, truth) == 1.0


def test_foreground_masks_uses_the_absolute_value():
    foreground = np.array([[0.5], [-0.5], [0.05], [0.0]])  # 4 pixels, 1 frame
    masks = foreground_masks(foreground, frame_height=2, frame_width=2, threshold=0.1)
    assert masks.shape == (1, 2, 2)
    assert masks[0].tolist() == [[True, True], [False, False]]


def test_resize_masks_keeps_the_majority():
    mask = np.zeros((1, 4, 4), dtype=bool)
    mask[0, :2, :2] = True  # top-left 2 x 2 block
    small = resize_masks(mask, frame_height=2, frame_width=2)
    assert small[0].tolist() == [[True, False], [False, False]]


def test_temporal_median_ignores_a_short_visit():
    matrix = np.full((3, 5), 0.5)
    matrix[1, 2] = 1.0  # a "dancer" on pixel 2 in frame 3 only
    background, foreground = temporal_median(matrix)
    assert np.allclose(background, 0.5)
    assert foreground[1, 2] == 0.5


def test_frame_numbers_are_evenly_spaced():
    assert frame_numbers_to_annotate(400, 10)[0] == 1
    assert frame_numbers_to_annotate(400, 10)[-1] == 400
    assert len(frame_numbers_to_annotate(400, 10)) == 10


def test_evaluation_skips_without_masks(tmp_path):
    masks = {"method": np.zeros((3, 2, 2), dtype=bool)}
    assert evaluate_if_possible(masks, tmp_path, 2, 2) is None


def test_ground_truth_mask_is_loaded_and_resized(tmp_path):
    drawn = np.zeros((8, 8), dtype=np.uint8)
    drawn[:4, :4] = 255
    Image.fromarray(drawn).save(tmp_path / "frame_0003_mask.png")
    truth = load_ground_truth(tmp_path, frame_height=2, frame_width=2)
    assert list(truth) == [3]
    assert truth[3].tolist() == [[True, False], [False, False]]


def test_mask_iou():
    first = np.array([True, True, False, False])
    second = np.array([True, False, True, False])
    assert mask_iou(first, second) == 1 / 3
    assert mask_iou(first, first) == 1.0


def test_describe_mask_file_counts_gray_pixels(tmp_path):
    drawn = np.zeros((4, 6), dtype=np.uint8)
    drawn[0, :3] = 255
    drawn[1, 0] = 128  # one anti-aliased edge pixel
    Image.fromarray(drawn).save(tmp_path / "frame_0001_mask.png")
    report = describe_mask_file(tmp_path / "frame_0001_mask.png")
    assert (report["width"], report["height"]) == (6, 4)
    assert report["gray_pixels"] == 1
    assert report["dancer_share"] == 4 / 24


def test_keep_region_clears_the_outside():
    masks = np.ones((2, 4, 5), dtype=bool)
    kept = keep_region(masks, (1, 3, 2, 4))
    assert kept.dtype == bool
    assert kept.sum() == 2 * 2 * 2
    assert kept[:, 1:3, 2:4].all()
    assert keep_region(masks, None).sum() == masks.sum()
