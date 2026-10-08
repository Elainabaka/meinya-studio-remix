"""
AI high-band restoration (Apollo) for sources a lossy codec has cut off.

A 128 kbps MP3 has nothing above ~16 kHz; at 0.9x that wall lands at 14.4 kHz, well inside hearing. The model
rebuilds the missing band, and ONLY that band is used: the result is `source + highpass(model(source) - source)`
from 1 kHz under the detected cutoff. Reason (measured 03/10/2026 on 8 songs): the model also rewrites the band
that is already there (4-16 kHz, at -16..-20 dB) without getting closer to the reference, and on full-band sources
it removes 1.7-4.6 dB of the top band. So a source without a codec wall is left alone (`source_cutoff` is None).

The model treats every channel as mono, so it runs on Mid/Side: content it adds to L and R separately is
decorrelated (delta L/R correlation ~0 against 0.3-0.8 for the songs) and would smear centred sibilance.

Optional: `torch` + `omegaconf`, the model source in `core/apollo/` (github.com/JusperLee/Apollo, CC BY-SA 4.0)
and its weights in `.studio_cache/models/apollo.bin`. Check `restore_available()` first.
"""

import contextlib
import hashlib
import importlib.util
import io
import os
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from scipy import signal

from .audio_io import save_audio
from .deesser import _rms_envelope
from .stems import _resample, fit_length

MODEL_NAME = "apollo-fp32-ms"         # part of the cache key: change it when the delta would come out different
MODEL_RATE = 44100
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "apollo")
WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(MODEL_DIR)), ".studio_cache", "models", "apollo.bin")
WEIGHTS_URL = "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin"
WEIGHTS_SHA256 = "99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76"

FULL_BAND_HZ = 19500.0                # a cutoff from here up is inaudible even at 0.8x: nothing to restore
WALL_DB = 20.0                        # a codec lowpass drops >= 26 dB within 1.5 kHz; a natural roll-off ~13 dB
MARGIN_HZ = 1000.0                    # the delta starts this far under the cutoff
CHUNK_S, PAD_S, OVERLAP_S = 6.0, 1.0, 1.0   # the model sees +-0.54 s around a frame, so 1 s of context is exact
_model = []


def restore_available() -> bool:
    """Cheap (no torch import): the libraries, the model source and the weights are all there."""
    try:
        if any(importlib.util.find_spec(name) is None for name in ("torch", "omegaconf")):
            return False
    except (ImportError, ValueError):
        return False
    return os.path.isfile(os.path.join(MODEL_DIR, "apollo.py")) and os.path.isfile(WEIGHTS)


def source_cutoff(audio: np.ndarray, sample_rate: int) -> Optional[float]:
    """Frequency (Hz) of a codec lowpass in the long-term spectrum, or None when the source is full band.
    It is the first frequency above 10 kHz that sits 30 dB under the 8-12 kHz level, and it must be a wall."""
    if sample_rate < 44100 or audio.shape[1] < 16384:
        return None
    freqs, power = signal.welch(np.mean(audio, axis=0), sample_rate, nperseg=16384, noverlap=0)
    db = 10.0 * np.log10(power + 1e-30)
    level = db[(freqs > 8000) & (freqs < 12000)].mean()
    below = freqs[(freqs > 10000) & (db < level - 30.0)]
    if not len(below) or below[0] >= FULL_BAND_HZ:
        return None
    cutoff = float(below[0])
    wall = db[(freqs > cutoff - 1500) & (freqs < cutoff - 1000)].mean() - db[(freqs > cutoff) & (freqs < cutoff + 500)].mean()
    return cutoff if wall >= WALL_DB else None


def _load_model() -> Callable[[np.ndarray], np.ndarray]:
    if not _model:
        import torch
        from collections import defaultdict
        from typing import Any
        from omegaconf import DictConfig, ListConfig
        from omegaconf.base import ContainerMetadata, Metadata
        from omegaconf.nodes import AnyNode
        from torch.torch_version import TorchVersion
        from .apollo.apollo import Apollo

        with open(WEIGHTS, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != WEIGHTS_SHA256:
                raise RuntimeError(f"Apollo weights do not match the pinned SHA-256: {WEIGHTS}")
        allowed = [int, list, dict, defaultdict, Any, ContainerMetadata, Metadata, AnyNode, ListConfig, DictConfig,
                   TorchVersion]
        with torch.serialization.safe_globals(allowed):
            checkpoint = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        with contextlib.redirect_stdout(io.StringIO()):   # the constructor prints its band layout
            net = Apollo(**dict(checkpoint["model_args"]))
        net.load_state_dict(checkpoint["state_dict"], strict=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net.eval().to(device)

        def run(chunk: np.ndarray) -> np.ndarray:
            with torch.inference_mode():
                return net(torch.from_numpy(np.ascontiguousarray(chunk))[None].to(device)).cpu().numpy()[0]
        _model.append(run)
    return _model[0]


def _chunked(model: Callable[[np.ndarray], np.ndarray], x: np.ndarray, rate: int,
             progress_callback: Optional[Callable[[str, float], None]] = None) -> np.ndarray:
    """model(x) in 6 s chunks, each with 1 s of context on both sides, cross-faded over 1 s."""
    n = x.shape[1]
    size, over, pad = round(CHUNK_S * rate), round(OVERLAP_S * rate), round(PAD_S * rate)
    starts = list(range(0, max(1, n - over), size - over))
    total, weights = np.zeros_like(x), np.zeros(n, np.float32)
    for i, start in enumerate(starts):
        end = min(n, start + size)
        lo = max(0, min(start - pad, n - size - 2 * pad))
        context = x[:, lo:min(n, lo + size + 2 * pad)]
        if context.shape[1] < size + 2 * pad:
            context = np.pad(context, ((0, 0), (0, size + 2 * pad - context.shape[1])))
        out = model(context)[:, start - lo:end - lo]
        w = np.ones(end - start, np.float32)
        fade = min(over, end - start)
        if i:
            w[:fade] = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        if i < len(starts) - 1:
            w[-fade:] = np.linspace(1.0, 0.0, fade, dtype=np.float32)
        total[:, start:end] += out * w
        weights[start:end] += w
        if progress_callback:
            progress_callback("Restoring Highs (AI)...", (i + 1) / len(starts))
    return total / np.maximum(weights, 1e-6)[None]


def restore_delta(
    audio: np.ndarray,
    sample_rate: int,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Optional[np.ndarray]:
    """What to add to `audio` (channels, samples) to fill the band its codec removed; None = nothing to restore.
    `model` maps (channels, samples) at 44.1 kHz to the same shape (default: Apollo)."""
    cutoff = source_cutoff(audio, sample_rate)
    if cutoff is None:
        return None
    stereo = audio.shape[0] == 2
    x = np.stack([(audio[0] + audio[1]) / 2, (audio[0] - audio[1]) / 2]) if stereo else audio
    peak = float(np.max(np.abs(x)))
    if peak < 1e-9:
        return None
    if progress_callback:
        progress_callback("Loading Restore Model (AI)...", 0.0)
    scale = min(1.0, 0.8 / peak)
    x = np.ascontiguousarray(_resample(x * scale, sample_rate, MODEL_RATE), dtype=np.float32)
    residual = (_chunked(model or _load_model(), x, MODEL_RATE, progress_callback) - x) / scale
    residual = fit_length(_resample(residual, MODEL_RATE, sample_rate), audio.shape[1])
    if stereo:
        residual = np.stack([residual[0] + residual[1], residual[0] - residual[1]])
    sos = signal.butter(6, cutoff - MARGIN_HZ, btype="highpass", fs=sample_rate, output="sos")
    delta = signal.sosfiltfilt(sos, residual, axis=-1)
    # Digital silence stays silent: the model leaves a faint noise floor everywhere
    envelope = _rms_envelope(audio, sample_rate, cutoff_hz=10.0).mean(axis=0)
    lo, hi = 10.0 ** (-90 / 20), 10.0 ** (-75 / 20)
    return np.ascontiguousarray(delta * np.clip((envelope - lo) / (hi - lo), 0.0, 1.0)[None], dtype=np.float32)


def restore_cache_file(file_path: str, sample_rate: int, cache_dir: str) -> str:
    st = os.stat(file_path)
    key = f"{os.path.abspath(file_path)}|{st.st_size}|{st.st_mtime_ns}|{MODEL_NAME}|{sample_rate}"
    return os.path.join(cache_dir, hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".flac")


def restore_cached(file_path: str, sample_rate: int, cache_dir: str) -> bool:
    return os.path.isfile(restore_cache_file(file_path, sample_rate, cache_dir))


def cached_restore(
    file_path: str,
    audio: np.ndarray,
    sample_rate: int,
    cache_dir: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Optional[np.ndarray]:
    """restore_delta once per (file, size, mtime, model); later calls load the FLAC cache."""
    if source_cutoff(audio, sample_rate) is None:
        return None
    path = restore_cache_file(file_path, sample_rate, cache_dir)
    if os.path.isfile(path):
        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            if sr == sample_rate and data.shape == (audio.shape[1], audio.shape[0]):
                return np.ascontiguousarray(data.T)
        except (OSError, RuntimeError):
            pass  # a truncated cache: run the model again
    delta = restore_delta(audio, sample_rate, progress_callback)
    if delta is not None:
        os.makedirs(cache_dir, exist_ok=True)
        save_audio(path, delta, sample_rate, subtype="PCM_24")
    return delta
