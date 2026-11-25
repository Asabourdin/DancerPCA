"""Figures, side-by-side montages, GIF/MP4 export"""

from pathlib import Path

import numpy as np
from PIL import Image


def save_frame_image(frame: np.ndarray, output_path: Path, upscale_factor: int = 1) -> None:
    """Save one grayscale frame (values in [0, 1]) as a PNG file"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pixel_values = np.clip(frame, 0.0, 1.0) * 255
    image = Image.fromarray(pixel_values.astype(np.uint8))
    new_size = (image.width * upscale_factor, image.height * upscale_factor) #upscale_factor makes the image bigger so it is easier to look at
    image = image.resize(new_size, Image.Resampling.NEAREST)
    image.save(output_path)
