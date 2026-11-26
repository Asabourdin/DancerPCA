"""Part 2: the known offset between two videos, from the music only (not the images)

We take the sound of each window, measure how loud it is every 10 ms (the amplitude envelope),
and slide one envelope along the other to find where they match best
"""

import subprocess
from pathlib import Path

import numpy as np
import yt_dlp

from rpca_dance.rhythm import find_rhythm_offset
from rpca_dance.video import check_ffmpeg_installed, is_url, video_name


def download_audio(url: str, output_dir: Path) -> Path:
    """Download the sound track of a video with yt-dlp, unless it is already downloaded

    The video files only hold the images, so the sound is downloaded on its own, as <id>_audio.<ext>
    Output: the path of the audio file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    options = {
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": str(output_dir / "%(id)s_audio.%(ext)s"),
        "quiet": True,
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(options) as downloader:
        info = downloader.extract_info(url, download=False)
        audio_path = Path(downloader.prepare_filename(info))
        if audio_path.exists():
            print(f"Audio already downloaded: {audio_path}")
            return audio_path
        print("Downloading the audio...")
        downloader.download([url])
    return audio_path


def load_audio_window(audio_path: Path, start_seconds: float, duration_seconds: float, sample_rate: int) -> np.ndarray:
    """Cut a time window of the sound with ffmpeg, in mono at a low sample rate

    ffmpeg writes 16-bit samples to its output, which we read directly (no file)
    Output: a float array of samples between -1 and 1
    """
    check_ffmpeg_installed()
    command = [
        "ffmpeg", "-loglevel", "error",
        "-ss", str(start_seconds),
        "-t", str(duration_seconds),
        "-i", str(audio_path),
        "-ac", "1",  # mono
        "-ar", str(sample_rate),
        "-f", "s16le", "-",
    ]
    raw_bytes = subprocess.run(command, check=True, capture_output=True).stdout
    return np.frombuffer(raw_bytes, dtype=np.int16) / 32768


def amplitude_envelope(samples: np.ndarray, sample_rate: int, envelope_rate: int) -> np.ndarray:
    """How loud the sound is over time: mean of |sample| over blocks of 1 / envelope_rate seconds

    Example: at 8000 samples per second and envelope_rate 100, each value is the mean of 80 samples (10 ms)
    """
    block_size = sample_rate // envelope_rate
    number_of_blocks = len(samples) // block_size
    blocks = np.abs(samples[:number_of_blocks * block_size]).reshape(number_of_blocks, block_size)
    return blocks.mean(axis=1)


def envelope_offset(
    reference_envelope: np.ndarray, learner_envelope: np.ndarray, max_lag: int, min_overlap: int
) -> tuple[int, np.ndarray, np.ndarray]:
    """Lag (in envelope values) where the two envelopes match best, between -max_lag and +max_lag

    Learner value t is paired with reference value t + lag, as for the motion curves, and the match is
    the correlation of the paired values (same function as the rhythm offset search)
    Output: the best lag, the lags tried, and the correlation of each
    """
    return find_rhythm_offset(reference_envelope, learner_envelope, 0, max_lag, min_overlap)


def window_envelope(
    source: str, raw_audio_dir: Path, start_seconds: float, duration_seconds: float, sample_rate: int, envelope_rate: int
) -> np.ndarray:
    """Envelope of the sound of one video window; source is a YouTube link or a local video file with sound"""
    audio_path = Path(source)
    if is_url(source):
        audio_path = download_audio(source, raw_audio_dir)
    samples = load_audio_window(audio_path, start_seconds, duration_seconds, sample_rate)
    print(f"Audio of {video_name(source)}: {len(samples) / sample_rate:.1f} s")
    return amplitude_envelope(samples, sample_rate, envelope_rate)


def audio_offset_frames(lag: int, envelope_rate: int, frames_per_second: int) -> int:
    """Turn a lag in envelope values into a lag in video frames, rounded to the nearest frame"""
    return int(round(lag * frames_per_second / envelope_rate))
