"""
Audio I/O Module: Robust loading and exporting across formats (WAV, MP3, FLAC, M4A, OGG)
using SoundFile with automatic FFmpeg transcoding fallback.
"""

import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from typing import Tuple


def load_audio(file_path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
    """
    Load an audio file into a 2D numpy array of shape (channels, samples) with float32 type.
    Converts mono to stereo if needed.
    Supports MP3, WAV, FLAC, M4A, AAC, OGG, OPUS, and video containers.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    # First attempt: soundfile direct read
    try:
        data, sr = sf.read(file_path, dtype='float32', always_2d=True)
        # sf returns shape (samples, channels), we standardize to (channels, samples)
        audio = data.T
    except Exception:
        # Fallback: transcode to temp WAV using FFmpeg
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        temp_wav.close()
        try:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", file_path,
                "-vn",  # no video
                "-acodec", "pcm_f32le",
                temp_wav.name
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            data, sr = sf.read(temp_wav.name, dtype='float32', always_2d=True)
            audio = data.T
        finally:
            if os.path.exists(temp_wav.name):
                try:
                    os.remove(temp_wav.name)
                except OSError:
                    pass

    # Ensure stereo (2 channels)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite, nonempty samples")
    if audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)
    elif audio.shape[0] > 2:
        # Downmix multi-channel to stereo
        audio = audio[:2, :]

    # Sample Rate Guard: Standardize low-res audio (< 44.1kHz) to 44.1kHz for DSP filter stability
    min_sr = 44100
    effective_target_sr = target_sr if target_sr is not None else (min_sr if sr < min_sr else sr)

    if sr != effective_target_sr:
        from math import gcd
        from scipy import signal
        g = gcd(int(effective_target_sr), int(sr))
        up, down = int(effective_target_sr) // g, int(sr) // g
        audio = signal.resample_poly(audio, up, down, axis=1).astype(np.float32)
        sr = effective_target_sr

    return audio.astype(np.float32), sr


def save_audio(
    file_path: str,
    audio: np.ndarray,
    sample_rate: int,
    bitrate: str = "320k",
    subtype: str = "PCM_24"
) -> str:
    """Write a completed file atomically; an encoder failure preserves any previous output."""
    if audio.ndim not in (1, 2) or audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("Audio must contain finite, nonempty samples")
    folder = os.path.dirname(os.path.abspath(file_path))
    os.makedirs(folder, exist_ok=True)
    fd, partial = tempfile.mkstemp(suffix=os.path.splitext(file_path)[1], prefix=".audio_", dir=folder)
    os.close(fd)
    try:
        _write_audio(partial, audio, sample_rate, bitrate, subtype)
        os.replace(partial, file_path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return file_path


def _write_audio(
    file_path: str,
    audio: np.ndarray,
    sample_rate: int,
    bitrate: str = "320k",
    subtype: str = "PCM_24"
) -> str:
    """
    Save audio (shape: channels, samples) to file.
    Supports .wav, .flac, .mp3, .ogg.
    Uses FFmpeg for MP3 encoding to guarantee pristine high-bitrate output.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    ext = os.path.splitext(file_path)[1].lower()

    # Audio must be (samples, channels) for soundfile write
    if audio.ndim == 1:
        out_data = audio[:, np.newaxis]
    elif audio.ndim == 2:
        out_data = audio.T
    else:
        out_data = audio

    # Clip to avoid hard digital wrap-around
    out_data = np.clip(out_data, -1.0, 1.0)

    if ext in [".wav", ".flac", ".ogg"]:
        if ext in (".wav", ".flac"):
            sf.write(file_path, out_data, sample_rate, subtype=subtype)
        else:
            sf.write(file_path, out_data, sample_rate)
        return file_path

    # For MP3 or others, write temp wav and encode via FFmpeg
    temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_wav.close()
    try:
        sf.write(temp_wav.name, out_data, sample_rate, subtype="PCM_24")
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", temp_wav.name,
            "-codec:a", "libmp3lame" if ext == ".mp3" else "copy",
            "-b:a", bitrate,
            file_path
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        if os.path.exists(temp_wav.name):
            try:
                os.remove(temp_wav.name)
            except OSError:
                pass

    return file_path
