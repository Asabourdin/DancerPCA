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
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
TO_ANNOTATE_DIR = GROUND_TRUTH_DIR / "to_annotate"
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

# Largest time shift (in frames) tried when aligning the two videos
MAX_OFFSET_FRAMES = 40

# Set to an integer to force the offset instead of searching for it
MANUAL_OFFSET_FRAMES = None


# Lighting stress test (stage 5): artificial brightness changes added to M
# Slow ramp: the gain goes from 1 - amplitude (first frame) to 1 + amplitude (last frame)
LIGHTING_RAMP_AMPLITUDE = 0.2
# Sudden step, like a phone's auto-exposure: frames after this one get the gain below
LIGHTING_STEP_FRAME = 100
LIGHTING_STEP_GAIN = 0.8
