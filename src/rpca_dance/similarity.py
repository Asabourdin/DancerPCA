"""Part 2: compare the silhouettes of a learner with the reference, frame by frame (IoU)

Steps: crop the masks to the box where the dancers are, resize them to a common size,
find the time offset between the two videos, then compute the IoU of each pair of frames
"""

import numpy as np

from rpca_dance.masks import resize_masks


def dancer_box(masks: np.ndarray, outlier_share: float) -> tuple[int, int, int, int]:
    """Box that holds the dancers over the whole video: (top, bottom, left, right), bottom and right excluded

    We take all dancer pixels of all frames and drop the `outlier_share` most extreme ones on each side
    (percentiles), so a few stray pixels far from the group do not make the box huge
    """
    _, rows, columns = np.nonzero(masks)  # the frame index is not needed
    low = 100 * outlier_share
    high = 100 * (1 - outlier_share)
    top, bottom = np.percentile(rows, [low, high])
    left, right = np.percentile(columns, [low, high])
    return int(top), int(bottom) + 1, int(left), int(right) + 1


def comparison_frame_range(
    reference_length: int, learner_length: int, start_offset: int, max_shift: int
) -> tuple[int, int]:
    """Reference frames that can be paired with a learner frame for some offset of the search

    Output: (first index, last index excluded), clipped to the reference video
    """
    first = max(start_offset - max_shift, 0)
    last = min(start_offset + max_shift + learner_length, reference_length)
    return first, last


def normalize_silhouettes(
    masks: np.ndarray, box: tuple[int, int, int, int], width: int, height: int
) -> np.ndarray:
    """Crop every mask to the dancer box, then resize it to height x width

    Both cameras are fixed but not placed the same way: after this step, the group fills the same frame
    box: (top, bottom, left, right), from dancer_box
    Output: boolean array of shape (T, height, width)
    """
    top, bottom, left, right = box
    cropped = masks[:, top:bottom, left:right]
    return resize_masks(cropped, height, width)


def flip_left_right(masks: np.ndarray) -> np.ndarray:
    """Mirror every mask (left and right swapped), as in a mirrored dance cover"""
    return masks[:, :, ::-1]


def iou_per_frame(first_masks: np.ndarray, second_masks: np.ndarray) -> np.ndarray:
    """IoU of each pair of frames: |A and B| / |A or B|, for all frames at once

    Inputs: two boolean arrays of shape (T, height, width)
    Output: T values between 0 (no overlap) and 1 (same silhouette); 1 when both masks are empty
    """
    intersections = np.sum(first_masks & second_masks, axis=(1, 2))
    unions = np.sum(first_masks | second_masks, axis=(1, 2))
    scores = np.ones(len(unions))
    has_dancers = unions > 0
    scores[has_dancers] = intersections[has_dancers] / unions[has_dancers]
    return scores


def overlapping_frames(
    reference: np.ndarray, learner: np.ndarray, offset: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair learner frame t with reference frame t + offset, and keep the pairs that exist in both videos

    Output: the reference frames, the learner frames, and the reference frame numbers (1 = first frame)
    """
    learner_indices = np.arange(len(learner))
    reference_indices = learner_indices + offset
    inside = (reference_indices >= 0) & (reference_indices < len(reference))
    return reference[reference_indices[inside]], learner[learner_indices[inside]], reference_indices[inside] + 1


def find_offset(
    reference: np.ndarray, learner: np.ndarray, start_offset: int, max_shift: int, min_overlap: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Try every offset from start_offset - max_shift to start_offset + max_shift and keep the best mean IoU

    Offsets that leave fewer than min_overlap pairs of frames are skipped (mean IoU set to -1)
    Output: the best offset, the offsets tried, and the mean IoU of each
    """
    offsets = np.arange(start_offset - max_shift, start_offset + max_shift + 1)
    mean_scores = np.full(len(offsets), -1.0)
    for position, offset in enumerate(offsets):
        reference_part, learner_part, _ = overlapping_frames(reference, learner, int(offset))
        if len(learner_part) >= min_overlap:
            mean_scores[position] = iou_per_frame(reference_part, learner_part).mean()
    best_offset = int(offsets[np.argmax(mean_scores)])
    return best_offset, offsets, mean_scores


def similarity_score(
    reference: np.ndarray, learner: np.ndarray, start_offset: int, max_shift: int,
    min_overlap: int, manual_offset: int | None,
) -> dict:
    """Align the two videos in time, then score each pair of frames with the IoU

    Inputs: normalized silhouettes (see normalize_silhouettes); if manual_offset is not None, it is used as is
    Output: a dictionary with "offset", "frame_numbers" (reference frames), "scores" (IoU per frame),
    "mean_score", and the alignment search: "offsets_tried", "mean_by_offset"
    """
    offsets_tried = np.array([])
    mean_by_offset = np.array([])
    offset = manual_offset
    if manual_offset is None:
        offset, offsets_tried, mean_by_offset = find_offset(reference, learner, start_offset, max_shift, min_overlap)
    reference_part, learner_part, frame_numbers = overlapping_frames(reference, learner, offset)
    scores = iou_per_frame(reference_part, learner_part)
    return {
        "offset": offset,
        "frame_numbers": frame_numbers,
        "scores": scores,
        "mean_score": float(scores.mean()),
        "offsets_tried": offsets_tried,
        "mean_by_offset": mean_by_offset,
    }


def score_with_and_without_flip(
    reference: np.ndarray, learner: np.ndarray, start_offset: int, max_shift: int,
    min_overlap: int, manual_offset: int | None,
) -> tuple[dict, dict]:
    """Score the learner as filmed and mirrored (left and right swapped)

    Output: the result of similarity_score for the learner as filmed, then for the mirrored learner
    """
    as_filmed = similarity_score(reference, learner, start_offset, max_shift, min_overlap, manual_offset)
    mirrored = similarity_score(reference, flip_left_right(learner), start_offset, max_shift, min_overlap, manual_offset)
    return as_filmed, mirrored


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Average of each value with its neighbours over `window` frames (same length as the input)

    Near the start and the end the window is cut: we divide by the number of frames it really holds
    """
    window_sums = np.convolve(values, np.ones(window), mode="same")
    frames_in_window = np.convolve(np.ones(len(values)), np.ones(window), mode="same")
    return window_sums / frames_in_window


def lowest_moments(
    values: np.ndarray, frame_numbers: np.ndarray, count: int, min_gap: int
) -> list[tuple[int, float]]:
    """The `count` lowest values, at least `min_gap` frames apart

    Output: a list of (frame number, value), lowest first
    """
    moments = []
    for index in np.argsort(values):
        chosen_frames = np.array([frame for frame, _ in moments])
        if len(moments) == 0 or np.min(np.abs(chosen_frames - frame_numbers[index])) >= min_gap:
            moments.append((int(frame_numbers[index]), float(values[index])))
        if len(moments) == count:
            break
    return moments


def biggest_drops(scores: np.ndarray, frame_numbers: np.ndarray, count: int, window: int) -> list[tuple[int, float]]:
    """The `count` moments where the score is lowest, at least `window` frames apart

    The scores are first averaged over `window` frames (one second at 8 fps), so one bad frame is not a "moment"
    Output: a list of (reference frame number, smoothed score), lowest first
    """
    smoothed = moving_average(scores, window)
    return lowest_moments(smoothed, frame_numbers, count, window)


def iou_chance_level(
    reference: np.ndarray, learner: np.ndarray, start_offset: int, max_shift: int, min_overlap: int,
    repeats: int, seed: int, percentile: float,
) -> tuple[float, np.ndarray]:
    """Chance level of the pixel score: the same offset search, with the learner frames in a random order

    Each repeat shuffles the learner frames and keeps the best mean IoU over all offsets,
    so we compare "best of all offsets" with "best of all offsets"
    Output: the given percentile of these best values, and all of them
    """
    random_generator = np.random.default_rng(seed)
    best_values = np.zeros(repeats)
    for repeat in range(repeats):
        shuffled = learner[random_generator.permutation(len(learner))]
        _, _, mean_by_offset = find_offset(reference, shuffled, start_offset, max_shift, min_overlap)
        best_values[repeat] = mean_by_offset.max()
    return float(np.percentile(best_values, percentile)), best_values


def distort_silhouettes(masks: np.ndarray, shift_pixels: int, erase_share: float, seed: int) -> np.ndarray:
    """A slightly wrong copy of the silhouettes, for a sanity check

    Every mask is moved `shift_pixels` to the right, then a random `erase_share` of its dancer pixels is removed
    """
    random_generator = np.random.default_rng(seed)
    shifted = np.roll(masks, shift_pixels, axis=2)
    keep = random_generator.random(masks.shape) >= erase_share
    return shifted & keep


def sanity_checks(
    reference: np.ndarray, time_shift: int, max_shift: int, min_overlap: int,
    shift_pixels: int, erase_share: float, seed: int,
) -> dict:
    """Three checks of the score, with copies of the reference silhouettes as the "learner"

    1. the reference against itself: the score must be 1 at offset 0
    2. a copy that starts time_shift frames later: the alignment must find the offset time_shift
    3. a distorted copy (distort_silhouettes): the score must drop below 1
    Output: {"self": result, "time shift": result, "distorted": result} (results of similarity_score)
    """
    shifted_copy = reference[time_shift:]
    distorted_copy = distort_silhouettes(reference, shift_pixels, erase_share, seed)
    return {
        "self": similarity_score(reference, reference, 0, max_shift, min_overlap, None),
        "time shift": similarity_score(reference, shifted_copy, 0, max_shift, min_overlap, None),
        "distorted": similarity_score(reference, distorted_copy, 0, max_shift, min_overlap, None),
    }
