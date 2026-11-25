"""Video preparation: download, cut, resize, grayscale, frames <-> matrix"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import yt_dlp
from PIL import Image


def check_ffmpeg_installed() -> None:
    """Stop with a clear message if ffmpeg is not installed"""
    #Troubleshoot for runs outside of my dev env
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg was not found: install it with: brew install ffmpeg"
        )


def is_url(source: str) -> bool:
    """Return True if the source looks like a web address, False for a local path"""
    #-> to manage download
    return source.startswith("http://") or source.startswith("https://")


def download_video(url: str, output_dir: Path) -> Path:
    """Download a video with yt-dlp, unless it is already dwnld

    Inputs: the video URL and the folder where the file goes
    Output: the path of the downloaded .mp4 file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {
        # Video only, at most 720p: we shrink it to 160 x 90 anyways
        "format": "bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]",
        "outtmpl": str(output_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
        video_path = Path(downloader.prepare_filename(info))
        if video_path.exists():
            print(f"Video already downloaded: {video_path}")
            return video_path
        print("Downloading the video...")
        downloader.download([url])
    return video_path


def extract_frames(
    video_path: Path,
    frames_dir: Path,
    start_seconds: float,
    duration_seconds: float,
    frames_per_second: int,
    frame_width: int,
    frame_height: int,
) -> None:
    """Cut a time window from the video and save it as small grayscale PNG frames

    ffmpeg does all the work: it keeps the window, samples the frames resizes them and converts them to gray. Old frames in frames_dir are deleted.
    """
    check_ffmpeg_installed()
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    video_filter = f"fps={frames_per_second},scale={frame_width}:{frame_height},format=gray"
    command = [
        "ffmpeg", "-loglevel", "error",
        "-ss", str(start_seconds),
        "-t", str(duration_seconds),
        "-i", str(video_path),
        "-vf", video_filter,
        str(frames_dir / "frame_%04d.png"),
    ]
    print("Extracting frames with ffmpeg...")
    subprocess.run(command, check=True)


def load_frames(frames_dir: Path) -> np.ndarray:
    """Load the PNG frames of a folder, in order

    Output: a float array of shape (T, height, width) with values in [0, 1]
    """
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir}")
    frames = []
    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("L")
        frames.append(np.asarray(image, dtype=np.float64) / 255.0)
    return np.stack(frames)


def prepare_video(
    source: str,
    frames_dir: Path,
    raw_video_dir: Path,
    start_seconds: float,
    duration_seconds: float,
    frames_per_second: int,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Go from a URL or a local file to an array of frames.
    Output: a float array of shape (T, height, width) with values in [0, 1].
    """
    if is_url(source):
        #if the source is a url then the first step is to download the video
        video_path = download_video(source, raw_video_dir)
    else:
        video_path = Path(source)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
    extract_frames(
        video_path, frames_dir, start_seconds, duration_seconds,
        frames_per_second, frame_width, frame_height,
    )
    return load_frames(frames_dir)


def frames_to_matrix(frames: np.ndarray) -> np.ndarray:
    """Turn frames of shape (T, height, width) into a matrix of shape (height * width, T)

    Each frame is flattened row by row into one column
    """
    number_of_frames = frames.shape[0]
    return frames.reshape(number_of_frames, -1).T


def matrix_to_frames(matrix: np.ndarray, frame_height: int, frame_width: int) -> np.ndarray:
    """Turn a matrix of shape (height * width, T) back into frames of shape (T, height, width)"""
    number_of_frames = matrix.shape[1]
    return matrix.T.reshape(number_of_frames, frame_height, frame_width)
