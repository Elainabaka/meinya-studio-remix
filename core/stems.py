"""
AI Stem Separation (Demucs htdemucs_ft) for stem-aware remixing.

Stems only STEER processing (reverb send, drum transient shaping, vocal de-essing); the dry path
stays the original mix and receives only the per-stem *differences*. Reason: the four estimated
stems sum back to the mix at only ~18 dB SNR (measured on real songs), so summing stems would put
separation artifacts straight into the listener's ears.

Optional dependency: `torch` + `demucs` (pip install demucs). Check `stems_available()` first.
"""

import hashlib
import os
from fractions import Fraction
from typing import Callable, Dict, Optional

import numpy as np
import soundfile as sf
from scipy import signal
from .audio_io import save_audio

MODEL_NAME = "htdemucs_ft"            # Demucs v4 hybrid transformer, fine-tuned bag (1 model per stem)
STEM_NAMES = ("drums", "bass", "other", "vocals")
_CACHE_SCALE = 0.25                   # 12 dB headroom so 24-bit FLAC cache never clips a hot stem
_models = {}


def stems_available() -> bool:
    try:
        import torch  # noqa: F401
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def stems_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def fit_length(audio: np.ndarray, num_samples: int) -> np.ndarray:
    """Crop or zero-pad (channels, samples) to exactly num_samples."""
    if audio.shape[1] >= num_samples:
        return audio[:, :num_samples]
    return np.pad(audio, ((0, 0), (0, num_samples - audio.shape[1])))


def _resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio
    ratio = Fraction(int(sr_to), int(sr_from))
    return signal.resample_poly(audio, ratio.numerator, ratio.denominator, axis=1).astype(np.float32)


def _load_model(name: str):
    if name not in _models:
        from demucs.pretrained import get_model
        model = get_model(name)
        model.eval()
        _models[name] = model
    return _models[name]


def separate_stems(
    audio: np.ndarray,
    sample_rate: int,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    model_name: str = MODEL_NAME,
) -> Dict[str, np.ndarray]:
    """Split a stereo mix (channels, samples) into drums/bass/other/vocals at the input sample rate."""
    import torch
    from demucs.apply import apply_model

    if progress_callback:
        progress_callback("Loading Stem Model (AI)...", 0.0)
    model = _load_model(model_name)
    wav = torch.from_numpy(np.ascontiguousarray(_resample(audio, sample_rate, model.samplerate), dtype=np.float32))
    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8

    def on_chunk(info: dict):
        if progress_callback and info.get("state") == "end":
            done = info["model_idx_in_bag"] + min(1.0, (info["segment_offset"] + 1) / info["audio_length"])
            progress_callback("Separating Stems (AI)...", min(1.0, done / info["models"]))

    with torch.no_grad():
        out = apply_model(
            model, ((wav - mean) / std)[None], device=stems_device(), shifts=1, split=True, overlap=0.25,
            callback=on_chunk, callback_arg={"audio_length": wav.shape[1]},
        )[0]
    out = (out * std + mean).cpu().numpy()
    return {
        name: fit_length(_resample(out[i], model.samplerate, sample_rate), audio.shape[1]).astype(np.float32)
        for i, name in enumerate(model.sources)
    }


def cached_stems(
    file_path: str,
    audio: np.ndarray,
    sample_rate: int,
    cache_dir: str,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, np.ndarray]:
    """Separate once per (file, size, mtime, model); later calls load the FLAC cache in ~1 s."""
    folder = stem_cache_folder(file_path, sample_rate, cache_dir)
    files = {name: os.path.join(folder, f"{name}.flac") for name in STEM_NAMES}
    if all(os.path.isfile(f) for f in files.values()):
        try:
            cached = {}
            for name, f in files.items():
                data, sr = sf.read(f, dtype="float32", always_2d=True)
                if sr != sample_rate or data.shape != (audio.shape[1], audio.shape[0]):
                    raise ValueError("Incomplete stem cache")
                cached[name] = data.T / _CACHE_SCALE
            return cached
        except (OSError, RuntimeError, ValueError):
            pass  # a truncated cache must not steer the mix; separate the original again
    stems = separate_stems(audio, sample_rate, progress_callback)
    os.makedirs(folder, exist_ok=True)
    for name, stem in stems.items():
        save_audio(files[name], stem * _CACHE_SCALE, sample_rate, subtype="PCM_24")
    return stems


def stem_cache_folder(file_path: str, sample_rate: int, cache_dir: str) -> str:
    st = os.stat(file_path)
    key = f"{os.path.abspath(file_path)}|{st.st_size}|{st.st_mtime_ns}|{MODEL_NAME}|{sample_rate}"
    return os.path.join(cache_dir, hashlib.sha1(key.encode("utf-8")).hexdigest()[:16])


def stems_cached(file_path: str, sample_rate: int, cache_dir: str) -> bool:
    folder = stem_cache_folder(file_path, sample_rate, cache_dir)
    return all(os.path.isfile(os.path.join(folder, f"{name}.flac")) for name in STEM_NAMES)
