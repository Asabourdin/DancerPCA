"""Part 2: motion rhythm score

At each frame we measure how much the dancers move. This gives one curve per video.
We compare the ups and downs of the two curves with a correlation, after aligning them in time
"""

import numpy as np

from rpca_dance.similarity import lowest_moments, moving_average, overlapping_frames


def motion_curve(silhouettes: np.ndarray) -> np.ndarray:
    """Share of pixels that switch between dancer and background from one frame to the next

    Input: boolean silhouettes of shape (T, height, width)
    Output: T - 1 values; value t is the change from frame t + 1 to frame t + 2 (frames counted from 1)
    """
    changed_pixels = silhouettes[1:] != silhouettes[:-1]
    return changed_pixels.mean(axis=(1, 2))


def sparse_motion_curve(sparse_frames: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Robustness check: the same idea with S itself, mean over the dancer box of | |S_t| - |S_t-1| |

    Input: S as frames of shape (T, height, width), and the dancer box (top, bottom, left, right)
    Output: T - 1 values
    """
    top, bottom, left, right = box
    dancer_strength = np.abs(sparse_frames[:, top:bottom, left:right])
    return np.abs(dancer_strength[1:] - dancer_strength[:-1]).mean(axis=(1, 2))


def standardize(values: np.ndarray) -> np.ndarray:
    """Subtract the mean and divide by the standard deviation (a flat curve stays at 0)"""
    spread = values.std()
    if spread == 0:
        return np.zeros(len(values))
    return (values - values.mean()) / spread


def curve_correlation(first_curve: np.ndarray, second_curve: np.ndarray) -> float:
    """Correlation of two curves of the same length: 1 = same ups and downs, 0 = no relation, -1 = opposite"""
    return float(np.mean(standardize(first_curve) * standardize(second_curve)))


def find_rhythm_offset(
    reference_curve: np.ndarray, learner_curve: np.ndarray, start_offset: int, max_shift: int, min_overlap: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Try every offset around start_offset and keep the one with the best correlation

    Learner value t is paired with reference value t + offset. Offsets with fewer than min_overlap
    pairs are skipped (correlation set to -1)
    Output: the best offset, the offsets tried, and the correlation of each
    """
    offsets = np.arange(start_offset - max_shift, start_offset + max_shift + 1)
    correlations = np.full(len(offsets), -1.0)
    for position, offset in enumerate(offsets):
        reference_part, learner_part, _ = overlapping_frames(reference_curve, learner_curve, int(offset))
        if len(learner_part) >= min_overlap:
            correlations[position] = curve_correlation(reference_part, learner_part)
    best_offset = int(offsets[np.argmax(correlations)])
    return best_offset, offsets, correlations


def rhythm_chance_level(
    reference_curve: np.ndarray, learner_curve: np.ndarray, start_offset: int, max_shift: int,
    min_overlap: int, min_circular_shift: int, percentile: float,
) -> tuple[float, np.ndarray]:
    """Chance level: shift the learner curve in a circle, run the same offset search, keep the best correlation

    The circular shift keeps the shape of the curve (its smoothness) but breaks its timing.
    Shifts go from min_circular_shift to length - min_circular_shift
    Output: the given percentile of the best correlations, and all of them
    """
    shifts = np.arange(min_circular_shift, len(learner_curve) - min_circular_shift + 1)
    best_values = np.zeros(len(shifts))
    for position, shift in enumerate(shifts):
        shifted_curve = np.roll(learner_curve, shift)
        _, _, correlations = find_rhythm_offset(reference_curve, shifted_curve, start_offset, max_shift, min_overlap)
        best_values[position] = correlations.max()
    return float(np.percentile(best_values, percentile)), best_values


def local_correlation(reference_part: np.ndarray, learner_part: np.ndarray, window: int) -> np.ndarray:
    """Correlation of the two aligned curves over a sliding window of `window` values

    Output: one value per window position (len - window + 1 values); value i covers values i to i + window - 1
    """
    number_of_windows = len(reference_part) - window + 1
    correlations = np.zeros(number_of_windows)
    for start in range(number_of_windows):
        correlations[start] = curve_correlation(reference_part[start:start + window], learner_part[start:start + window])
    return correlations


def autocorrelation(curve: np.ndarray, min_lag: int, max_lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Correlation of the curve with itself shifted by each lag from min_lag to max_lag

    A high value at some lag means the movement repeats after that many frames
    Output: the lags and their correlations
    """
    lags = np.arange(min_lag, max_lag + 1)
    correlations = np.zeros(len(lags))
    for position, lag in enumerate(lags):
        correlations[position] = curve_correlation(curve[:-lag], curve[lag:])
    return lags, correlations


def rhythm_score(
    reference_curve: np.ndarray, learner_curve: np.ndarray, start_offset: int, max_shift: int,
    min_overlap: int, local_window: int, number_of_moments: int,
) -> dict:
    """Align the two motion curves, then measure how well they follow each other, over the whole clip and locally

    The curves should already be smoothed (moving_average). Curve value t belongs to frame t + 2
    Output: "offset", "score" (best correlation), "offsets_tried", "correlation_by_offset",
    "reference_part", "learner_part", "frame_numbers" (reference frames of the aligned values),
    "local" (sliding correlation), "local_frame_numbers" (frame at the middle of each window), "lost_moments"
    """
    offset, offsets_tried, correlations = find_rhythm_offset(reference_curve, learner_curve, start_offset, max_shift, min_overlap)
    reference_part, learner_part, value_numbers = overlapping_frames(reference_curve, learner_curve, offset)
    frame_numbers = value_numbers + 1  # value t (counted from 1) is the change into frame t + 1
    local = local_correlation(reference_part, learner_part, local_window)
    local_frame_numbers = frame_numbers[local_window // 2:local_window // 2 + len(local)]
    return {
        "offset": offset,
        "score": float(correlations.max()),
        "offsets_tried": offsets_tried,
        "correlation_by_offset": correlations,
        "reference_part": reference_part,
        "learner_part": learner_part,
        "frame_numbers": frame_numbers,
        "local": local,
        "local_frame_numbers": local_frame_numbers,
        "lost_moments": lowest_moments(local, local_frame_numbers, number_of_moments, local_window),
    }


def smoothed_motion_curve(silhouettes: np.ndarray, smoothing_window: int) -> np.ndarray:
    """motion_curve followed by a moving average over smoothing_window frames"""
    return moving_average(motion_curve(silhouettes), smoothing_window)


def rhythm_sanity_checks(
    reference_silhouettes: np.ndarray, time_shift: int, max_shift: int, min_overlap: int,
    smoothing_window: int, distorted_silhouettes: np.ndarray,
) -> dict:
    """Same three checks as for the pixel score, on the rhythm score

    1. the reference against itself: score 1 at offset 0
    2. a copy that starts time_shift frames later: offset time_shift
    3. a distorted copy (given, see similarity.distort_silhouettes): score below 1
    Output: {"self": (offset, score), "time shift": (offset, score), "distorted": (offset, score)}
    """
    reference_curve = smoothed_motion_curve(reference_silhouettes, smoothing_window)
    copies = {
        "self": reference_curve,
        "time shift": reference_curve[time_shift:],
        "distorted": smoothed_motion_curve(distorted_silhouettes, smoothing_window),
    }
    checks = {}
    for check_name, copy_curve in copies.items():
        offset, _, correlations = find_rhythm_offset(reference_curve, copy_curve, 0, max_shift, min_overlap)
        checks[check_name] = (offset, float(correlations.max()))
    return checks


def passage_selection(
    number_of_frames: int, passages_seconds: list[tuple[float, float]], window_start_seconds: float, frames_per_second: int
) -> np.ndarray:
    """Which reference frames fall inside the chosen passages (for example the unison passages)

    Frame k (counted from 1) is at window_start_seconds + (k - 1) / frames_per_second in the video.
    A passage (start, end) keeps the frames with start <= time < end
    Output: boolean array of number_of_frames values; value k - 1 is True when frame k is kept
    """
    frame_times = window_start_seconds + np.arange(number_of_frames) / frames_per_second
    selection = np.zeros(number_of_frames, dtype=bool)
    for start_seconds, end_seconds in passages_seconds:
        selection = selection | ((frame_times >= start_seconds) & (frame_times < end_seconds))
    return selection


def selected_pairs(
    reference_curve: np.ndarray, learner_curve: np.ndarray, offset: int, frame_selection: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair the two curves at a given offset, then keep only the pairs whose reference frame is selected

    frame_selection: boolean array over the reference frames (value k - 1 for frame k), from passage_selection
    Output: the reference values, the learner values, and their reference frame numbers
    """
    reference_part, learner_part, value_numbers = overlapping_frames(reference_curve, learner_curve, offset)
    frame_numbers = value_numbers + 1  # value t (counted from 1) is the change into frame t + 1
    keep = frame_selection[frame_numbers - 1]
    return reference_part[keep], learner_part[keep], frame_numbers[keep]


def selected_correlation(
    reference_curve: np.ndarray, learner_curve: np.ndarray, offset: int, frame_selection: np.ndarray
) -> float:
    """Rhythm correlation at a known offset (no search), on the selected frames only"""
    reference_part, learner_part, _ = selected_pairs(reference_curve, learner_curve, offset, frame_selection)
    return curve_correlation(reference_part, learner_part)


def selected_chance_level(
    reference_curve: np.ndarray, learner_curve: np.ndarray, offset: int, frame_selection: np.ndarray,
    min_circular_shift: int, percentile: float,
) -> tuple[float, np.ndarray]:
    """Chance level at a known offset: shift the whole learner curve in a circle, then correlate on the same frames

    There is no search, so there is no maximum to take: each shift gives one correlation
    Output: the given percentile of these correlations, and all of them
    """
    shifts = np.arange(min_circular_shift, len(learner_curve) - min_circular_shift + 1)
    correlations = np.zeros(len(shifts))
    for position, shift in enumerate(shifts):
        shifted_curve = np.roll(learner_curve, shift)
        correlations[position] = selected_correlation(reference_curve, shifted_curve, offset, frame_selection)
    return float(np.percentile(correlations, percentile)), correlations


def passage_coverage(
    reference_curve: np.ndarray, learner_curve: np.ndarray, offset: int, passage_selections: list[np.ndarray]
) -> list[tuple[int, int]]:
    """How much of each passage the learner video covers at the given offset

    Output: one (covered values, values in the passage) pair per passage; the first frame has no motion value
    """
    coverage = []
    for frame_selection in passage_selections:
        _, _, frame_numbers = selected_pairs(reference_curve, learner_curve, offset, frame_selection)
        coverage.append((len(frame_numbers), int(frame_selection[1:].sum())))
    return coverage
