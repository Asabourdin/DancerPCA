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


# CORTIS GO dance practice video
REFERENCE_VIDEO_URL = "https://www.youtube.com/watch?v=gSm-eTRl9JE"

# Time window with a fixed camera: from 0:24 to 1:14 
START_SECONDS = 24.0
DURATION_SECONDS = 50.0

# Sampling and frame size: 50 s at 8 fps gives 400 frames
FRAMES_PER_SECOND = 8
FRAME_WIDTH = 160
FRAME_HEIGHT = 90


# Robust PCA params
# -> Stopping rule for the constrained version: ||M - L - S||_F / ||M||_F < tol
RPCA_TOLERANCE = 1e-7
RPCA_MAX_ITERATIONS = 500

# Penalized version: maximum number of alternating steps
PENALIZED_MAX_ITERATIONS = 200

# Multipliers of the default lambda = 1 / sqrt(max(n, T)) tried on the video
LAMBDA_MULTIPLIERS = [0.5, 1.0, 2.0]

# Simulations
RANDOM_SEED = 0
SIMULATION_SIZE = 100
SIMULATION_SUCCESS_TOLERANCE = 1e-3

# Masks and evaluation
# A pixel belongs to a dancer when |S| is above this value (pixels are in [0, 1])
MASK_THRESHOLD = 0.1
USE_OTSU_THRESHOLD = False

# Number of frames exported for hand annotation, and their upscaling factor
NUMBER_OF_FRAMES_TO_ANNOTATE = 10
ANNOTATION_UPSCALE_FACTOR = 4


# similarity score
# Common size (width, height) of the cropped silhouettes.
SILHOUETTE_WIDTH = 64
SILHOUETTE_HEIGHT = 64

# Largest time shift (in frames) tried when aligning the two videos.
MAX_OFFSET_FRAMES = 40

# Set to an integer to force the offset instead of searching for it.
MANUAL_OFFSET_FRAMES = None
