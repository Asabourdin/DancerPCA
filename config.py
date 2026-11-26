"""All parameters of the project in one place.

Paths are built from the location of this file, so they work the same
from main.py, from the notebook and from the tests.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_VIDEO_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
GROUND_TRUTH_DIR = PROJECT_ROOT / "ground_truth"  # hand-made masks, tracked by git
TO_ANNOTATE_DIR = DATA_DIR / "to_annotate"  # video frames, never committed
RESULTS_DIR = PROJECT_ROOT / "results"
DECOMPOSITIONS_DIR = RESULTS_DIR / "decompositions"


# CORTIS GO dance practice video
REFERENCE_VIDEO_URL = "https://www.youtube.com/watch?v=gSm-eTRl9JE"

# Time window with a fixed camera: from 0:24 to 1:14 
START_SECONDS = 24.0
DURATION_SECONDS = 50.0

# Sampling and frame size: 50 s at 8 fps gives 400 frames
FRAMES_PER_SECOND = 8
FRAME_WIDTH = 160
FRAME_HEIGHT = 90

# Color frames (4 times larger): the frames to annotate by hand, and the size of the hand-made masks
COLOR_FRAME_WIDTH = 640
COLOR_FRAME_HEIGHT = 360

# Second resolution, only for the resolution comparison of stage 5
HIGH_RES_FRAME_WIDTH = 320
HIGH_RES_FRAME_HEIGHT = 180


# Robust PCA params (proximal gradient descent on the penalized problem)
# -> Stop when the objective drops by less than this (relative) between two iterations
RPCA_TOLERANCE = 1e-7
RPCA_MAX_ITERATIONS = 2000

# Toy example (4 pixels, 3 frames): weights lambda1 (low rank) and lambda2 (sparse)
TOY_LOW_RANK_WEIGHT = 0.1
TOY_SPARSE_WEIGHT = 0.05

# Video: two rules to choose lambda1 and lambda2 (see rpca_dance/decomposition.py)
# "noise": both from the noise level sigma (Lasso chapter)
# "candes": lambda1 from the noise level, lambda2 = lambda1 / sqrt(max(n, T))
LAMBDA_RULES = ["noise", "candes"]
# Rule used for the masks, the baselines comparison and part 2 (to confirm with the F1 scores)
MAIN_LAMBDA_RULE = "candes"
# Both lambdas are multiplied by these values for the comparison
LAMBDA_MULTIPLIERS = [0.5, 1.0, 2.0]

# FISTA (accelerated proximal gradient): its objective can go up for a step,
# so it stops when ||L_new - L_old||_F / ||L_new||_F < this value
FISTA_TOLERANCE = 1e-5

# Simulations (phase transition diagram), noiseless, n x n matrices
RANDOM_SEED = 0
SIMULATION_SIZE = 100
SIMULATION_RANK_RATIOS = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # r / n
SIMULATION_SPARSITIES = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]  # rho
SIMULATION_TRIALS = 3
# lambda1 for the simulations; lambda2 = lambda1 / sqrt(n)
SIMULATION_LOW_RANK_WEIGHT = 0.1
# Success: ||L_hat - L0||_F < factor * lambda1 * sqrt(r) and rank(L_hat) = r
SIMULATION_SUCCESS_FACTOR = 3
SENSITIVITY_FACTORS = [2, 3, 5]

# SDP check with CVXPY
SDP_CHECK_SIZE = 30
SDP_CHECK_RANK = 2
SDP_CHECK_SPARSITY = 0.05
SDP_TIMING_SIZES = [20, 40, 60, 80, 100]
# Tight tolerance so that our solver and CVXPY are compared at their real optimum
SDP_CHECK_TOLERANCE = 1e-9
SDP_CHECK_MAX_ITERATIONS = 20000

# Masks and evaluation
# A pixel belongs to a dancer when |S| is above this value (pixels are in [0, 1])
MASK_THRESHOLD = 0.1
USE_OTSU_THRESHOLD = False

# Number of frames exported for hand annotation (color frames, 640 x 360)
NUMBER_OF_FRAMES_TO_ANNOTATE = 10

# Baselines
CLASSIC_PCA_RANK = 1


# similarity score
# Common size (width, height) of the  silhouettes
SILHOUETTE_WIDTH = 64
SILHOUETTE_HEIGHT = 64

# The silhouette box drops this share of the most extreme dancer pixels on each side (stray pixels)
SILHOUETTE_OUTLIER_SHARE = 0.01

# Time alignment: learner frame t is compared with reference frame t + offset
# Known offset: each learner window is chosen by ear so that it starts at the same moment of the song as the
# reference window, so the offset is 0. The version without search uses it as is; the search starts from it
LEARNER_START_OFFSET_FRAMES = 0
# Largest time shift (in frames) tried around the starting offset: 40 frames = 5 s
MAX_OFFSET_FRAMES = 40
# An offset must leave at least this many pairs of frames to be considered
MIN_OVERLAP_FRAMES = 80

# Set to an integer to force the offset instead of searching for it
MANUAL_OFFSET_FRAMES = None


# Lighting stress test (stage 5): artificial brightness changes added to M
# Slow ramp: the gain goes from 1 - amplitude (first frame) to 1 + amplitude (last frame)
LIGHTING_RAMP_AMPLITUDE = 0.2
# Sudden step, like a phone's auto-exposure: frames after the step frame get the gain below
# Two positions: early (frame 100) and in the middle of the clip (frame 200)
LIGHTING_STEP_FRAMES = [100, 200]
LIGHTING_STEP_GAIN = 0.8

# Learner videos (part 2): dance covers of the same choreography, filmed with a fixed camera
# start_seconds is the moment that matches the start of the reference window (0:24), checked by ear
# region_of_interest: (top, bottom, left, right) at 160 x 90, bottom and right excluded, or None for the whole
# frame; the motion is measured only inside it (chosen by looking at the video, before any score)
LEARNER_VIDEOS = {
    # Group cover in a gym, 0:19 to 0:45 (the video ends there)
    "group": {
        "url": "https://www.youtube.com/watch?v=cPjUB5R3EpI",
        "start_seconds": 19.0,
        "duration_seconds": 26.0,
        "region_of_interest": None,
    },
    # Group cover on an outdoor stage, 0:20 to 1:10: the full 50 s of the reference window
    # The region leaves out the passers-by on the left, the right edge and the bunting at the bottom
    "outdoor": {
        "url": "https://www.youtube.com/watch?v=b8tOGBS7Zrw",
        "start_seconds": 20.0,
        "duration_seconds": 50.0,
        "region_of_interest": (28, 74, 46, 146),
    },
}
# Learner used by main.py when no --learner is given
DEFAULT_LEARNER = "group"

# Unison passages of the choreography (all members do the same moves), in seconds of the reference video,
# chosen from the choreography before looking at any score: 0:26 to 0:36 and 0:46 to 0:53
UNISON_PASSAGES_SECONDS = [(26.0, 36.0), (46.0, 53.0)]

# Camera drift check: median image of each block of this many frames (2 s), compared with the first block
DRIFT_BLOCK_FRAMES = 16

# Score report: the moments with the lowest score, averaged over one second, at least one second apart
NUMBER_OF_DROPS = 3
DROP_WINDOW_FRAMES = 8

# Sanity checks of the score (until or besides a learner video)
SANITY_TIME_SHIFT_FRAMES = 12  # 1.5 s
SANITY_SHIFT_PIXELS = 1  # on the 64 x 64 silhouettes
SANITY_ERASE_SHARE = 0.05

# Chance level of the pixel score: learner frames shuffled this many times, same offset search each time
PIXEL_CHANCE_REPEATS = 100

# Motion rhythm score (definition fixed before seeing any result)
# Moving average over this many frames, the same for both videos
RHYTHM_SMOOTHING_FRAMES = 3
# Chance level: circular shifts of the learner curve of at least this many frames (2 s)
RHYTHM_MIN_CIRCULAR_SHIFT_FRAMES = 16
# All chance levels: this percentile of the chance values (best value over the search, or the value at the known offset)
CHANCE_PERCENTILE = 95
# Sliding window for the local correlation (2 s); the lowest moments are where the rhythm is lost
RHYTHM_LOCAL_WINDOW_FRAMES = 16

# Known offset from the music (stage 6 follow-up): the sound of each window, mono, at a low sample rate
AUDIO_SAMPLE_RATE = 8000
# Amplitude envelope: one value every 10 ms (100 values per second)
AUDIO_ENVELOPE_RATE = 100
# The offset is searched within +-2 s (200 envelope values), with at least 10 s of overlap
AUDIO_MAX_LAG_VALUES = 200
AUDIO_MIN_OVERLAP_VALUES = 1000
