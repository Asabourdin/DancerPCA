"""Run the whole project with one command

python main.py                       part 1 on the reference video, then the part 2 sanity checks
python main.py --learner outdoor     the same, then the similarity scores of a learner from config.py
python main.py --learner <url or path> --learner-start 20 --learner-duration 50 --learner-region 28 74 46 146
Long experiments only run with their option: --simulations, --lambda-comparison, --resolution-comparison, --lighting-test
All parameters are in config.py. Every output goes to results/
"""

import time

import config
from rpca_dance.pipeline import (
    build_argument_parser, learner_settings, run_baselines, run_lambda_comparison, run_lighting_test_step,
    run_part_one, run_part_two, run_resolution_comparison, run_sanity_checks, run_simulations, sanity_silhouettes,
)

arguments = build_argument_parser(config).parse_args()
start_time = time.perf_counter()
is_reference = arguments.video == config.REFERENCE_VIDEO_URL

if arguments.simulations:
    print("\n=== Simulations and SDP check ===")
    run_simulations(config)

print("\n=== Part 1: Robust PCA on the video ===")
part_one = run_part_one(arguments.video, config)

print("\n=== Baselines: classic PCA and temporal median ===")
run_baselines(part_one, is_reference, config)

if arguments.lambda_comparison:
    print("\n=== Lambda comparison ===")
    run_lambda_comparison(arguments.video, part_one, is_reference, config)

if arguments.resolution_comparison:
    print("\n=== Resolution comparison (320 x 180) ===")
    run_resolution_comparison(arguments.video, part_one, is_reference, config)

if arguments.lighting_test:
    print("\n=== Lighting stress test ===")
    if is_reference:
        run_lighting_test_step(part_one, config)
    else:
        print("Skipped: the lighting test is scored with the hand-made masks of the reference video.")

print("\n=== Part 2: sanity checks of the scores ===")
run_sanity_checks(sanity_silhouettes(part_one["masks"], config), config)

if arguments.learner is not None:
    print("\n=== Part 2: similarity score of the learner ===")
    learner = learner_settings(arguments.learner, arguments.learner_start, arguments.learner_duration,
                               arguments.learner_region, config)
    run_part_two(arguments.video, part_one, learner, arguments.learner_offset, config)

print(f"\nDone in {time.perf_counter() - start_time:.0f} s. Figures and saved runs are in {config.RESULTS_DIR}")
