"""Baselines (not part of the course): classic PCA, temporal median, pretrained DeepLabV3"""

import time

import numpy as np
import torch
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

# Index of the "person" class among the 21 Pascal VOC classes used by DeepLabV3
PERSON_CLASS = 15


def classic_pca(matrix: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    """Classic PCA as a background model: keep the best rank-r approximation of M

    Truncated SVD: L = sum of the first `rank` singular triplets, S = M - L
    It is the best fit in the least-squares sense, so a large outlier pulls L towards it ("blends" into the background)
    Output: L and S
    """
    left_vectors, singular_values, right_vectors_transposed = np.linalg.svd(matrix, full_matrices=False)
    low_rank_part = (left_vectors[:, :rank] * singular_values[:rank]) @ right_vectors_transposed[:rank]
    return low_rank_part, matrix - low_rank_part


def temporal_median(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Background = the median of each pixel over time; foreground = M - background

    Each dancer covers a pixel less than half of the time, so the median gives the empty studio
    Output: L (the median column repeated T times) and S = M - L
    """
    median_background = np.median(matrix, axis=1, keepdims=True)
    low_rank_part = np.repeat(median_background, matrix.shape[1], axis=1)
    return low_rank_part, matrix - low_rank_part


def choose_device() -> torch.device:
    """Use Apple's GPU (mps) if it is available, otherwise the CPU"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_deeplab_model(device: torch.device) -> torch.nn.Module:
    """Load DeepLabV3 MobileNetV3-Large, pretrained on COCO with the 21 Pascal VOC classes

    The weights (about 45 MB) are downloaded by torchvision the first time
    """
    weights = DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT
    model = deeplabv3_mobilenet_v3_large(weights=weights)
    model.eval()
    return model.to(device)


def predict_person_batch(model: torch.nn.Module, color_frames: np.ndarray, device: torch.device) -> np.ndarray:
    """Run DeepLabV3 on a batch of color frames of shape (B, height, width, 3), uint8

    Output: boolean masks of shape (B, height, width), True where the most likely class is "person"
    """
    images = torch.from_numpy(color_frames).float().div(255).permute(0, 3, 1, 2)
    # Normalize with the ImageNet mean and standard deviation, like during training
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = ((images - mean) / std).to(device)
    with torch.no_grad():
        class_scores = model(images)["out"]
    predicted_classes = class_scores.argmax(dim=1).cpu().numpy()
    return predicted_classes == PERSON_CLASS


def deeplab_person_masks(color_frames: np.ndarray, batch_size: int) -> tuple[np.ndarray, float]:
    """Person masks for all frames with the pretrained DeepLabV3 model

    Input: color frames of shape (T, height, width, 3), uint8
    Output: boolean masks of shape (T, height, width), and the run time in seconds
    (model loading not included)
    """
    device = choose_device()
    model = load_deeplab_model(device)
    print(f"Running DeepLabV3 on {len(color_frames)} frames ({device})...")
    start_time = time.perf_counter()
    batch_masks = []
    for batch_start in range(0, len(color_frames), batch_size):
        batch = color_frames[batch_start:batch_start + batch_size]
        batch_masks.append(predict_person_batch(model, batch, device))
    run_time = time.perf_counter() - start_time
    return np.concatenate(batch_masks), run_time
