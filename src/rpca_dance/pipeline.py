"""The steps of main.py: each function runs one part of the project, prints what it finds and saves its figures

Every function takes `settings`, the config module, so all the parameters still come from config.py
"""

import argparse
from pathlib import Path
from types import ModuleType

import numpy as np

from rpca_dance.audio import (
    audio_offset_frames, audio_peak_check, envelope_offset, source_has_audio, window_envelope,
)
from rpca_dance.comparison import (
    comparison_silhouettes, known_offset_rhythm, motion_curves, passage_rhythm, pixel_comparison,
    prepare_learner, repetition_check, rhythm_comparison, sparse_motion_curves,
)
from rpca_dance.decomposition import decomposition_path, load_decomposition, run_decomposition
from rpca_dance.evaluation import evaluate_methods, load_ground_truth, precision_and_recall, print_f1_table
from rpca_dance.experiments import (
    all_settings, baseline_masks, lighting_gains, print_run_summary, rpca_masks_by_setting, run_lighting_test,
)
from rpca_dance.masks import foreground_masks, keep_region, resize_masks
from rpca_dance.plots import (
    build_montage_frames, plot_brightness_gains, plot_f1_scores, plot_frame_differences, plot_history,
    plot_mask_grid, plot_mean_f1_by_condition, plot_offset_search, plot_pixel_score, plot_rhythm,
    plot_solver_times, plot_success_maps, save_montage_gif, save_montage_image,
)
from rpca_dance.rhythm import rhythm_sanity_checks
from rpca_dance.sdp_check import compare_with_cvxpy, time_both_solvers
from rpca_dance.similarity import (
    biggest_drops, dancer_box, distort_silhouettes, normalize_silhouettes, sanity_checks,
)
from rpca_dance.simulation import load_results, run_simulation_grid, save_results, success_fraction
from rpca_dance.video import (
    camera_drift, check_fixed_camera, frames_to_matrix, load_frames, matrix_to_frames, prepare_video, video_name,
)

# Frames shown in the mask figures of the reference video (0-based): frames 45, 134, 267 and 356
SHOWN_FRAMES = [44, 133, 266, 355]


def build_argument_parser(settings: ModuleType) -> argparse.ArgumentParser:
    """The options of main.py; every default comes from config.py"""
    parser = argparse.ArgumentParser(description="Robust PCA on a dance video, and a similarity score for a learner")
    parser.add_argument("--video", default=settings.REFERENCE_VIDEO_URL, help="reference video: URL or local path")
    parser.add_argument("--learner", default=None,
                        help=f"learner video: URL, local path, or a name from config.py {list(settings.LEARNER_VIDEOS)}")
    parser.add_argument("--learner-start", type=float, default=None, help="start of the learner window (seconds)")
    parser.add_argument("--learner-duration", type=float, default=None, help="length of the learner window (seconds)")
    parser.add_argument("--learner-region", type=int, nargs=4, default=None, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"),
                        help="region of interest at 160 x 90 (bottom and right excluded)")
    parser.add_argument("--learner-offset", type=int, default=0,
                        help="known offset in frames, used when the audio offset is missing or not reliable")
    parser.add_argument("--simulations", action="store_true", help="SDP check and phase transition diagram")
    parser.add_argument("--lambda-comparison", action="store_true", help="both lambda rules, x0.5, x1 and x2")
    parser.add_argument("--resolution-comparison", action="store_true", help="Robust PCA at 320 x 180 too")
    parser.add_argument("--lighting-test", action="store_true", help="artificial brightness changes")
    return parser


def run_name(source: str, settings: ModuleType, frame_width: int, frame_height: int) -> str:
    """Name of the cached Robust PCA runs: "160x90" for the reference video, "<video name>_160x90" otherwise"""
    size = f"{frame_width}x{frame_height}"
    if source == settings.REFERENCE_VIDEO_URL:
        return size
    return f"{video_name(source)}_{size}"


def describe_camera(frames: np.ndarray, name: str, settings: ModuleType) -> None:
    """Fixed-camera check: biggest jump between two frames (a cut) and slow drift (a pan or tilt)"""
    check = check_fixed_camera(frames)
    drift = camera_drift(frames, settings.DRIFT_BLOCK_FRAMES)
    drift_sizes = np.hypot(drift[:, 0], drift[:, 1])
    print(f"Camera check ({name}): biggest jump {check['largest_jump']:.4f} at frame {check['largest_jump_frame']}, "
          f"{check['jump_ratio']:.2f} times the median jump; largest drift {drift_sizes.max():.2f} pixels")
    plot_frame_differences(check["frame_differences"], settings.FRAMES_PER_SECOND,
                           settings.RESULTS_DIR / f"{name}_camera_check_differences.png")


def run_part_one(source: str, settings: ModuleType) -> dict:
    """Part 1: prepare the video, check the camera, run Robust PCA (main rule), save the montage

    Output: {"frames", "result" (the Robust PCA run), "masks"}
    """
    width, height = settings.FRAME_WIDTH, settings.FRAME_HEIGHT
    frames = prepare_video(source, settings.FRAMES_DIR / "reference", settings.RAW_VIDEO_DIR, settings.START_SECONDS,
                           settings.DURATION_SECONDS, settings.FRAMES_PER_SECOND, width, height)
    print(f"Video ready: {frames.shape[0]} frames of {width} x {height} pixels")
    describe_camera(frames, "reference", settings)
    output_path = decomposition_path(settings.DECOMPOSITIONS_DIR, run_name(source, settings, width, height),
                                     settings.MAIN_LAMBDA_RULE, 1.0)
    result = run_decomposition(frames_to_matrix(frames), settings.MAIN_LAMBDA_RULE, 1.0, settings.FISTA_TOLERANCE,
                               settings.RPCA_MAX_ITERATIONS, output_path)
    masks = foreground_masks(result["sparse"], height, width, settings.MASK_THRESHOLD)
    print_run_summary(f"Robust PCA, {settings.MAIN_LAMBDA_RULE} rule", result, masks)
    montage = build_montage_frames(frames, matrix_to_frames(result["low_rank"], height, width),
                                   matrix_to_frames(result["sparse"], height, width), masks, 3.0)
    save_montage_image(montage, SHOWN_FRAMES, settings.RESULTS_DIR / "montage.png")
    save_montage_gif(montage, settings.FRAMES_PER_SECOND, settings.RESULTS_DIR / "montage.gif")
    plot_history(result, "objective", "objective F(L, S)", settings.RESULTS_DIR / "video_objective.png", log_scale=True)
    return {"frames": frames, "result": result, "masks": masks}


def report_f1(method_masks: dict, frame_height: int, frame_width: int, title: str, output_path: Path, settings: ModuleType) -> dict | None:
    """F1, precision and recall of each method against the hand-made masks; None if there are no masks

    Output: {method name: F1 per annotated frame}
    """
    truth = load_ground_truth(settings.GROUND_TRUTH_DIR, frame_height, frame_width)
    if len(truth) == 0:
        print(f"No hand-made masks in {settings.GROUND_TRUTH_DIR}: skipping the F1 scores.")
        return None
    scores = evaluate_methods(method_masks, truth)
    print_f1_table(scores, sorted(truth), title)
    for method_name, (precision, recall) in precision_and_recall(method_masks, truth).items():
        print(f"{method_name:<24s}precision {precision:.3f}, recall {recall:.3f}")
    plot_f1_scores(scores, title, output_path)
    return scores


def run_baselines(part_one: dict, is_reference: bool, settings: ModuleType) -> None:
    """Classic PCA and the temporal median next to Robust PCA: masks, run times, and F1 on the reference video"""
    height, width = settings.FRAME_HEIGHT, settings.FRAME_WIDTH
    methods = {"Robust PCA": part_one["masks"]}
    quick_masks, run_times = baseline_masks(frames_to_matrix(part_one["frames"]), height, width,
                                            settings.MASK_THRESHOLD, settings.CLASSIC_PCA_RANK)
    methods.update(quick_masks)
    run_times["Robust PCA"] = part_one["result"]["run_time"]
    for method_name, seconds in run_times.items():
        print(f"{method_name}: {seconds:.2f} s")
    plot_mask_grid(list(methods.values()), list(methods), SHOWN_FRAMES, settings.RESULTS_DIR / "baselines_comparison.png")
    if is_reference:
        report_f1(methods, height, width, "F1 against the hand-made masks, 160 x 90",
                  settings.RESULTS_DIR / "f1_scores_160x90.png", settings)


def run_lambda_comparison(source: str, part_one: dict, is_reference: bool, settings: ModuleType) -> None:
    """Both lambda rules, each multiplied by 0.5, 1 and 2: masks and F1 (option --lambda-comparison)"""
    height, width = settings.FRAME_HEIGHT, settings.FRAME_WIDTH
    masks, _ = rpca_masks_by_setting(
        frames_to_matrix(part_one["frames"]), all_settings(settings.LAMBDA_RULES, settings.LAMBDA_MULTIPLIERS),
        settings.DECOMPOSITIONS_DIR, run_name(source, settings, width, height), settings.FISTA_TOLERANCE,
        settings.RPCA_MAX_ITERATIONS, height, width, settings.MASK_THRESHOLD,
    )
    plot_mask_grid(list(masks.values()), list(masks), SHOWN_FRAMES, settings.RESULTS_DIR / "lambda_comparison.png")
    if is_reference:
        report_f1(masks, height, width, "F1 of every lambda setting, 160 x 90",
                  settings.RESULTS_DIR / "f1_lambda_comparison.png", settings)


def run_resolution_comparison(source: str, part_one: dict, is_reference: bool, settings: ModuleType) -> None:
    """Robust PCA (both rules) and the baselines at 320 x 180, F1, and the masks next to 160 x 90 (option)"""
    width, height = settings.HIGH_RES_FRAME_WIDTH, settings.HIGH_RES_FRAME_HEIGHT
    frames = prepare_video(source, settings.FRAMES_DIR / f"reference_{width}x{height}", settings.RAW_VIDEO_DIR,
                           settings.START_SECONDS, settings.DURATION_SECONDS, settings.FRAMES_PER_SECOND, width, height)
    matrix = frames_to_matrix(frames)
    masks, _ = rpca_masks_by_setting(
        matrix, all_settings(settings.LAMBDA_RULES, [1.0]), settings.DECOMPOSITIONS_DIR,
        run_name(source, settings, width, height), settings.FISTA_TOLERANCE, settings.RPCA_MAX_ITERATIONS,
        height, width, settings.MASK_THRESHOLD,
    )
    masks.update(baseline_masks(matrix, height, width, settings.MASK_THRESHOLD, settings.CLASSIC_PCA_RANK)[0])
    high_masks = masks[f"RPCA {settings.MAIN_LAMBDA_RULE} x1.0"]
    shrunk = resize_masks(high_masks, settings.FRAME_HEIGHT, settings.FRAME_WIDTH)
    overlap = np.sum(shrunk & part_one["masks"]) / np.sum(shrunk | part_one["masks"])
    print(f"IoU between the 160 x 90 masks and the 320 x 180 masks shrunk to 160 x 90: {overlap:.3f}")
    plot_mask_grid([part_one["masks"], high_masks], ["160 x 90", "320 x 180"], SHOWN_FRAMES,
                   settings.RESULTS_DIR / "resolution_comparison.png")
    if is_reference:
        report_f1(masks, height, width, "F1 against the hand-made masks, 320 x 180",
                  settings.RESULTS_DIR / "f1_scores_320x180.png", settings)


def run_lighting_test_step(part_one: dict, settings: ModuleType) -> None:
    """Artificial brightness ramp and steps: F1 of Robust PCA, classic PCA and the median (option --lighting-test)"""
    height, width = settings.FRAME_HEIGHT, settings.FRAME_WIDTH
    if len(load_ground_truth(settings.GROUND_TRUTH_DIR, height, width)) == 0:
        print(f"No hand-made masks in {settings.GROUND_TRUTH_DIR}: skipping the lighting test (it is scored with F1).")
        return
    matrix = frames_to_matrix(part_one["frames"])
    gains = lighting_gains(matrix.shape[1], settings.LIGHTING_RAMP_AMPLITUDE, settings.LIGHTING_STEP_FRAMES,
                           settings.LIGHTING_STEP_GAIN)
    plot_brightness_gains(gains, settings.RESULTS_DIR / "lighting_gains.png")
    by_change = {"original": {"Robust PCA": part_one["masks"]}}
    by_change["original"].update(baseline_masks(matrix, height, width, settings.MASK_THRESHOLD, settings.CLASSIC_PCA_RANK)[0])
    by_change.update(run_lighting_test(matrix, gains, part_one["result"], settings.DECOMPOSITIONS_DIR,
                                       settings.FISTA_TOLERANCE, settings.RPCA_MAX_ITERATIONS, height, width,
                                       settings.MASK_THRESHOLD, settings.CLASSIC_PCA_RANK))
    mean_scores = {}
    for change_name, masks in by_change.items():
        file_name = change_name.replace(" ", "_")
        scores = report_f1(masks, height, width, f"F1, {change_name}", settings.RESULTS_DIR / f"lighting_f1_{file_name}.png", settings)
        mean_scores[change_name] = {method_name: float(values.mean()) for method_name, values in scores.items()}
        plot_mask_grid(list(masks.values()), list(masks), SHOWN_FRAMES, settings.RESULTS_DIR / f"lighting_masks_{file_name}.png")
    plot_mean_f1_by_condition(mean_scores, settings.RESULTS_DIR / "lighting_mean_f1.png")


def simulation_grid(settings: ModuleType) -> dict:
    """Load the saved simulation grid (results/simulation_results.npz), or run it and save it (about 5 minutes)"""
    grid_path = settings.RESULTS_DIR / "simulation_results.npz"
    if grid_path.exists():
        print(f"Loading saved simulation grid: {grid_path.name}")
        return load_results(grid_path)
    results = run_simulation_grid(settings.SIMULATION_SIZE, settings.SIMULATION_RANK_RATIOS, settings.SIMULATION_SPARSITIES,
                                  settings.SIMULATION_TRIALS, settings.SIMULATION_LOW_RANK_WEIGHT,
                                  settings.FISTA_TOLERANCE, settings.RPCA_MAX_ITERATIONS, settings.RANDOM_SEED)
    save_results(results, grid_path)
    return results


def run_simulations(settings: ModuleType) -> None:
    """SDP check against CVXPY, run times, and the phase transition diagram (option --simulations)

    The simulation grid is saved in results/simulation_results.npz and loaded if it exists
    """
    check = compare_with_cvxpy(settings.SDP_CHECK_SIZE, settings.SDP_CHECK_RANK, settings.SDP_CHECK_SPARSITY,
                               settings.RANDOM_SEED, settings.SDP_CHECK_TOLERANCE, settings.SDP_CHECK_MAX_ITERATIONS)
    print(f"SDP check ({check['solver']}): relative gap between the objectives {check['objective_gap']:.1e}, "
          f"L differs by {check['low_rank_difference']:.1e}, S by {check['sparse_difference']:.1e}")
    timings = time_both_solvers(settings.SDP_TIMING_SIZES, settings.SDP_CHECK_RANK, settings.SDP_CHECK_SPARSITY,
                                settings.RANDOM_SEED, settings.SDP_CHECK_TOLERANCE, settings.SDP_CHECK_MAX_ITERATIONS)
    plot_solver_times(timings, settings.RESULTS_DIR / "sdp_timings.png")
    results = simulation_grid(settings)
    rpca_map, pca_map = success_fraction(results, settings.SIMULATION_LOW_RANK_WEIGHT, settings.SIMULATION_SUCCESS_FACTOR)
    plot_success_maps([rpca_map, pca_map], ["Robust PCA (FISTA)", "classic PCA (true rank given)"],
                      results["rank_ratios"], results["sparsities"], settings.RESULTS_DIR / "phase_transition.png")
    print(f"Phase transition: Robust PCA succeeds in {rpca_map.mean():.0%} of the trials, classic PCA in {pca_map.mean():.0%}")


def sanity_silhouettes(masks: np.ndarray, settings: ModuleType) -> np.ndarray:
    """Silhouettes of the reference video alone (dancer box over all frames), for the sanity checks"""
    box = dancer_box(masks, settings.SILHOUETTE_OUTLIER_SHARE)
    return normalize_silhouettes(masks, box, settings.SILHOUETTE_WIDTH, settings.SILHOUETTE_HEIGHT)


def run_sanity_checks(silhouettes: np.ndarray, settings: ModuleType) -> None:
    """The three checks of both scores: the reference against itself, a time-shifted copy, a distorted copy"""
    pixel_checks = sanity_checks(silhouettes, settings.SANITY_TIME_SHIFT_FRAMES, settings.MAX_OFFSET_FRAMES,
                                 settings.MIN_OVERLAP_FRAMES, settings.SANITY_SHIFT_PIXELS, settings.SANITY_ERASE_SHARE,
                                 settings.RANDOM_SEED)
    distorted = distort_silhouettes(silhouettes, settings.SANITY_SHIFT_PIXELS, settings.SANITY_ERASE_SHARE, settings.RANDOM_SEED)
    rhythm_checks = rhythm_sanity_checks(silhouettes, settings.SANITY_TIME_SHIFT_FRAMES, settings.MAX_OFFSET_FRAMES,
                                         settings.MIN_OVERLAP_FRAMES, settings.RHYTHM_SMOOTHING_FRAMES, distorted)
    for check_name, result in pixel_checks.items():
        rhythm_offset, rhythm_value = rhythm_checks[check_name]
        print(f"Sanity check, {check_name}: pixel score {result['mean_score']:.3f} at offset {result['offset']}, "
              f"rhythm score {rhythm_value:.3f} at offset {rhythm_offset}")


def learner_settings(
    learner: str, start_seconds: float | None, duration_seconds: float | None, region: list[int] | None, settings: ModuleType
) -> dict:
    """Window and region of the learner: from config.LEARNER_VIDEOS if `learner` is one of its names,
    otherwise `learner` is a URL or a path; the options given on the command line replace these values

    Output: {"name", "url", "start_seconds", "duration_seconds", "region_of_interest"}
    """
    chosen = {"name": video_name(learner), "url": learner, "start_seconds": 0.0,
              "duration_seconds": settings.DURATION_SECONDS, "region_of_interest": None}
    if learner in settings.LEARNER_VIDEOS:
        chosen.update(settings.LEARNER_VIDEOS[learner])
        chosen["name"] = learner
    if start_seconds is not None:
        chosen["start_seconds"] = start_seconds
    if duration_seconds is not None:
        chosen["duration_seconds"] = duration_seconds
    if region is not None:
        chosen["region_of_interest"] = tuple(region)
    return chosen


def prepare_learner_step(learner: dict, settings: ModuleType) -> dict:
    """Prepare the learner video exactly like the reference, check its camera, keep the region of interest

    Output: {"masks", "sparse"} (S as frames), both set to zero outside the region of interest
    """
    masks, sparse = prepare_learner(
        learner["url"], settings.FRAMES_DIR, settings.RAW_VIDEO_DIR, learner["start_seconds"], learner["duration_seconds"],
        settings.FRAMES_PER_SECOND, settings.FRAME_WIDTH, settings.FRAME_HEIGHT, settings.DECOMPOSITIONS_DIR,
        settings.MAIN_LAMBDA_RULE, settings.FISTA_TOLERANCE, settings.RPCA_MAX_ITERATIONS, settings.MASK_THRESHOLD,
    )
    frames = load_frames(settings.FRAMES_DIR / f"learner_{video_name(learner['url'])}")
    describe_camera(frames, f"learner_{learner['name']}", settings)
    region = learner["region_of_interest"]
    print(f"Region of interest of the learner: {region} (None = whole frame)")
    return {"masks": keep_region(masks, region), "sparse": keep_region(sparse, region)}


def audio_offset_check(reference_source: str, learner: dict, settings: ModuleType) -> dict:
    """Offset between the two windows from the music (amplitude envelopes), with the reliability check

    Output: the result of audio_peak_check, plus "frames" (the offset in video frames)
    """
    reference_envelope = window_envelope(reference_source, settings.RAW_VIDEO_DIR, settings.START_SECONDS,
                                         settings.DURATION_SECONDS, settings.AUDIO_SAMPLE_RATE, settings.AUDIO_ENVELOPE_RATE)
    learner_envelope = window_envelope(learner["url"], settings.RAW_VIDEO_DIR, learner["start_seconds"],
                                       learner["duration_seconds"], settings.AUDIO_SAMPLE_RATE, settings.AUDIO_ENVELOPE_RATE)
    _, lags, correlations = envelope_offset(reference_envelope, learner_envelope, settings.AUDIO_MAX_LAG_VALUES,
                                            settings.AUDIO_MIN_OVERLAP_VALUES)
    check = audio_peak_check(lags, correlations, settings.AUDIO_PEAK_DISTANCE_VALUES, settings.AUDIO_EDGE_MARGIN_VALUES,
                             settings.AUDIO_MIN_PEAK_RATIO)
    check["frames"] = audio_offset_frames(check["lag"], settings.AUDIO_ENVELOPE_RATE, settings.FRAMES_PER_SECOND)
    print(f"Audio offset: {check['lag'] / settings.AUDIO_ENVELOPE_RATE:+.2f} s ({check['frames']:+d} frames), "
          f"peak {check['best']:.3f}, next peak {check['next peak']:.3f}, ratio {check['ratio']:.2f}, "
          f"at the edge of the search: {check['at edge']}")
    return check


def known_offset(reference_source: str, learner: dict, user_offset: int, settings: ModuleType) -> int:
    """The known offset: from the audio when both videos have sound and the audio peak is reliable,
    otherwise the offset given with --learner-offset (default 0), with a warning
    """
    if not (source_has_audio(reference_source) and source_has_audio(learner["url"])):
        print(f"WARNING: a video has no sound, so the offset cannot come from the music. "
              f"Using --learner-offset = {user_offset} frames.")
        return user_offset
    check = audio_offset_check(reference_source, learner, settings)
    if check["reliable"]:
        print(f"The audio peak is reliable: known offset = {check['frames']} frames.")
        return check["frames"]
    print(f"WARNING: the audio peak is not reliable (it must be at least {settings.AUDIO_MIN_PEAK_RATIO} times the next "
          f"peak and not at the edge of the search). The rhythm score at a known offset needs a precise sync. "
          f"Using --learner-offset = {user_offset} frames.")
    return user_offset


def print_rhythm_result(title: str, result: dict, frames_per_second: int) -> None:
    """Score, offset and chance level; the lost moments only when the score beats its chance level"""
    above_chance = result["score"] > result["chance level"]
    print(f"{title}: {result['score']:.3f} at offset {result['offset']} frames, chance level {result['chance level']:.3f}, "
          f"above chance: {above_chance}")
    if not above_chance:
        print("  The score does not beat its chance level, so the moments where the rhythm is lost mean nothing here.")
        return
    for frame_number, value in result["lost_moments"]:
        print(f"  rhythm lost at {frame_number / frames_per_second:.1f} s of the reference window (local correlation {value:.2f})")


def run_search_version(reference_masks: np.ndarray, learner_masks: np.ndarray, name: str, settings: ModuleType) -> None:
    """Pixel score (as filmed and mirrored) and rhythm score, with an offset search and their chance levels"""
    start, shift = settings.LEARNER_START_OFFSET_FRAMES, settings.MAX_OFFSET_FRAMES
    reference_silhouettes, learner_silhouettes, _, _ = comparison_silhouettes(
        reference_masks, learner_masks, start, shift, settings.SILHOUETTE_OUTLIER_SHARE,
        settings.SILHOUETTE_WIDTH, settings.SILHOUETTE_HEIGHT,
    )
    pixel = pixel_comparison(reference_silhouettes, learner_silhouettes, start, shift, settings.MIN_OVERLAP_FRAMES,
                             settings.PIXEL_CHANCE_REPEATS, settings.RANDOM_SEED, settings.CHANCE_PERCENTILE)
    print(f"Pixel score: as filmed {pixel['as filmed']['mean_score']:.3f}, mirrored {pixel['mirrored']['mean_score']:.3f}, "
          f"kept {pixel['kept']}; chance level {pixel['chance level']:.3f}")
    kept = pixel[pixel["kept"]]
    drops = biggest_drops(kept["scores"], kept["frame_numbers"], settings.NUMBER_OF_DROPS, settings.DROP_WINDOW_FRAMES)
    reference_curve, learner_curve = motion_curves(reference_silhouettes, learner_silhouettes, settings.RHYTHM_SMOOTHING_FRAMES)
    rhythm = rhythm_comparison(reference_curve, learner_curve, start, shift, settings.MIN_OVERLAP_FRAMES,
                               settings.RHYTHM_MIN_CIRCULAR_SHIFT_FRAMES, settings.CHANCE_PERCENTILE,
                               settings.RHYTHM_LOCAL_WINDOW_FRAMES, settings.NUMBER_OF_DROPS)
    print_rhythm_result("Rhythm score with the offset search", rhythm, settings.FRAMES_PER_SECOND)
    plot_pixel_score(kept, pixel["chance level"], drops, settings.FRAMES_PER_SECOND, settings.RESULTS_DIR / f"pixel_score_{name}.png")
    plot_offset_search(pixel, rhythm, settings.RESULTS_DIR / f"offset_search_{name}.png")
    plot_rhythm(rhythm, settings.FRAMES_PER_SECOND, settings.RESULTS_DIR / f"rhythm_score_{name}_search.png")


def run_known_offset_version(reference: dict, learner: dict, offset: int, name: str, settings: ModuleType) -> None:
    """Rhythm score at the known offset (no search), on the whole clip and on the unison passages, and the |S| check

    reference and learner: {"masks", "sparse"} (S as frames)
    """
    reference_silhouettes, learner_silhouettes, reference_box, learner_box = comparison_silhouettes(
        reference["masks"], learner["masks"], offset, 0, settings.SILHOUETTE_OUTLIER_SHARE,
        settings.SILHOUETTE_WIDTH, settings.SILHOUETTE_HEIGHT,
    )
    reference_curve, learner_curve = motion_curves(reference_silhouettes, learner_silhouettes, settings.RHYTHM_SMOOTHING_FRAMES)
    rhythm = known_offset_rhythm(reference_curve, learner_curve, offset, settings.MIN_OVERLAP_FRAMES,
                                 settings.RHYTHM_MIN_CIRCULAR_SHIFT_FRAMES, settings.CHANCE_PERCENTILE,
                                 settings.RHYTHM_LOCAL_WINDOW_FRAMES, settings.NUMBER_OF_DROPS)
    print_rhythm_result("Rhythm score at the known offset", rhythm, settings.FRAMES_PER_SECOND)
    plot_rhythm(rhythm, settings.FRAMES_PER_SECOND, settings.RESULTS_DIR / f"rhythm_score_{name}_known_offset.png")
    unison = passage_rhythm(reference_curve, learner_curve, offset, settings.UNISON_PASSAGES_SECONDS, settings.START_SECONDS,
                            settings.FRAMES_PER_SECOND, settings.RHYTHM_MIN_CIRCULAR_SHIFT_FRAMES, settings.CHANCE_PERCENTILE)
    print(f"Unison passages: {unison['score']:.3f}, chance level {unison['chance level']:.3f}; "
          f"motion values covered per passage {unison['coverage']}")
    reference_sparse_curve, learner_sparse_curve = sparse_motion_curves(
        reference["sparse"], learner["sparse"], reference_box, learner_box, settings.RHYTHM_SMOOTHING_FRAMES,
    )
    check = known_offset_rhythm(reference_sparse_curve, learner_sparse_curve, offset, settings.MIN_OVERLAP_FRAMES,
                                settings.RHYTHM_MIN_CIRCULAR_SHIFT_FRAMES, settings.CHANCE_PERCENTILE,
                                settings.RHYTHM_LOCAL_WINDOW_FRAMES, settings.NUMBER_OF_DROPS)
    print(f"Robustness check with |S| (not used to decide): {check['score']:.3f}, chance level {check['chance level']:.3f}")
    lag, value = repetition_check(reference_curve, settings.RHYTHM_MIN_CIRCULAR_SHIFT_FRAMES, len(reference_curve) // 3)
    print(f"Repetition of the reference movement: strongest at a lag of {lag / settings.FRAMES_PER_SECOND:.1f} s ({value:.2f})")


def run_part_two(reference_source: str, part_one: dict, learner: dict, user_offset: int, settings: ModuleType) -> None:
    """Part 2 with a learner video: both versions of the score"""
    height, width = settings.FRAME_HEIGHT, settings.FRAME_WIDTH
    reference = {"masks": part_one["masks"], "sparse": matrix_to_frames(part_one["result"]["sparse"], height, width)}
    print(f"Learner: {learner['url']}, from {learner['start_seconds']} s for {learner['duration_seconds']} s")
    learner_parts = prepare_learner_step(learner, settings)
    offset = known_offset(reference_source, learner, user_offset, settings)
    run_known_offset_version(reference, learner_parts, offset, learner["name"], settings)
    run_search_version(reference["masks"], learner_parts["masks"], learner["name"], settings)
