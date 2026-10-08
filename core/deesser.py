"""
Split-band De-esser (relative detection, zero-phase, offline lookahead)
Tames harsh "s / x / ch" consonants that pitch-up (Nightcore / Sped Up) and air boosts exaggerate.
Only the band above `freq_hz` is turned down, and only while that band dominates the full-band
signal, so hi-hats and air in non-sibilant passages are left alone. Detection is relative
(band vs full band), so it behaves the same at any input level.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d


@dataclass
class DeEsserConfig:
    max_reduction_db: float = 6.0   # Deepest cut on a sibilant (0 = off)
    freq_hz: float = 5500.0         # Sibilance band starts here
    threshold_db: float = -8.0      # Band level vs full band where reduction starts (full-mix detector)
    vocal_threshold_db: float = -5.0  # Same, for a vocal-stem detector (a solo voice is naturally brighter)
    ratio: float = 3.0              # Above threshold: every `ratio` dB over -> (ratio-1)/ratio dB cut
    hold_ms: float = 12.0           # Keeps the cut steady across one consonant


def _highpass(x: np.ndarray, sample_rate: int, freq_hz: float) -> np.ndarray:
    sos = signal.butter(4, freq_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1)


def _rms_envelope(x: np.ndarray, sample_rate: int, cutoff_hz: float = 60.0) -> np.ndarray:
    sos = signal.butter(2, cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    return np.sqrt(np.maximum(signal.sosfiltfilt(sos, x * x), 0.0))


def sibilance_gain(
    detector: np.ndarray, sample_rate: int, config: DeEsserConfig, threshold_db: Optional[float] = None
) -> np.ndarray:
    """Per-sample linear gain (<= 1) for the sibilance band, detected on `detector` (channels, samples).
    Calibrated on real songs at 1.25x: -8 dB (mix) / -5 dB (vocal stem) acts on ~2-12 % of the time,
    p99 cut ~3 dB, i.e. only on the consonants."""
    threshold = config.threshold_db if threshold_db is None else threshold_db
    mono = np.mean(detector, axis=0)
    env_band = _rms_envelope(_highpass(mono, sample_rate, config.freq_hz), sample_rate)
    env_full = _rms_envelope(mono, sample_rate)

    rel_db = 20.0 * np.log10((env_band + 1e-9) / (env_full + 1e-9))
    gr_db = np.clip((rel_db - threshold) * (1.0 - 1.0 / config.ratio), 0.0, config.max_reduction_db)
    gr_db[env_full < 1e-4] = 0.0  # below -80 dBFS: nothing to de-ess

    hold = max(1, int(config.hold_ms * 0.001 * sample_rate))
    gr_db = maximum_filter1d(gr_db, size=hold)
    sos_smooth = signal.butter(1, 40.0, btype="lowpass", fs=sample_rate, output="sos")
    gr_db = np.maximum(signal.sosfiltfilt(sos_smooth, gr_db), 0.0)
    return (10.0 ** (-gr_db / 20.0)).astype(np.float32)


def apply_deesser(
    audio: np.ndarray,
    sample_rate: int,
    config: Optional[DeEsserConfig] = None,
    source: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Turns down the sibilance band of `source` inside `audio`.
    source=None: classic full-mix de-esser (source is the mix itself).
    source=vocal stem: only the vocal's sibilance is reduced; hi-hats in the mix are untouched.
    """
    if config is None:
        config = DeEsserConfig()
    if config.max_reduction_db <= 0.01 or audio.shape[1] < 1000:
        return audio

    src = audio if source is None else source
    threshold = config.threshold_db if source is None else config.vocal_threshold_db
    gain = sibilance_gain(src, sample_rate, config, threshold)
    band = _highpass(src, sample_rate, config.freq_hz)
    return (audio - band * (1.0 - gain)[np.newaxis, :]).astype(np.float32)
