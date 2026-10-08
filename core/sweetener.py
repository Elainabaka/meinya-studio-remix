"""
Psychoacoustic Sweetener & Auto Fine-Tune Engine
Implements Fletcher-Munson contour compensation, surgical de-mudding,
air sheen, zero-phase Mid/Side Mono-Bass focusing, and harmonic tape warmth.
"""

from dataclasses import dataclass
import numpy as np
from scipy import signal
import pedalboard

from .channels import per_channel


@dataclass
class SweetenerConfig:
    sub_cut_hz: float = 28.0            # Infrasonic filter: cleans amp headroom
    bass_boost_hz: float = 75.0         # Deep chest punch & body
    bass_boost_db: float = 2.5
    de_mud_hz: float = 350.0            # Removes "cardboard / boxy" frequency buildup
    de_mud_db: float = -2.0
    harshness_cut_hz: float = 3500.0    # Anti-fatigue notch for in-ear headphones
    harshness_cut_db: float = -1.5
    air_sheen_hz: float = 12500.0       # Silky high-end sparkle
    air_sheen_db: float = 2.0
    mono_bass_cutoff_hz: float = 115.0  # Frequencies below this become 100% Mono
    stereo_width: float = 1.25          # Side channel stereo expansion factor
    warmth_drive: float = 0.04          # Subtle tape harmonic saturation
    tape_flutter_depth: float = 0.0     # Analog tape reel wow & flutter depth (e.g. 0.0004)


def _apply_mid_side_processing(
    audio: np.ndarray,
    sample_rate: int,
    mono_cutoff_hz: float = 115.0,
    stereo_width: float = 1.25
) -> np.ndarray:
    """
    Drives sub-bass to 100% Mono (zero phase cancellation on phones/clubs)
    and expands high-frequency stereo width on Side channels.
    """
    if audio.shape[0] < 2:
        return audio

    L = audio[0]
    R = audio[1]

    mid = 0.5 * (L + R)
    side = 0.5 * (L - R)

    # Zero-phase highpass on side channel to eliminate stereo sub-bass
    if mono_cutoff_hz > 10.0 and side.shape[0] > 100:
        sos = signal.butter(4, mono_cutoff_hz, btype='highpass', fs=sample_rate, output='sos')
        side = signal.sosfiltfilt(sos, side)

    # Scale side channel for stereo widening
    side = side * stereo_width

    L_out = mid + side
    R_out = mid - side

    return np.stack([L_out, R_out], axis=0).astype(np.float32)


def _apply_harmonic_warmth(audio: np.ndarray, drive: float = 0.04) -> np.ndarray:
    """
    Applies gentle analog tape-style soft clipping to generate pleasing 2nd & 3rd harmonics.
    """
    if drive <= 0.001:
        return audio
    
    alpha = 1.0 + drive * 2.5
    # Smooth hyperbolic tangent saturation normalized to unity gain
    saturated = np.tanh(audio * alpha) / np.tanh(alpha)
    return saturated.astype(np.float32)


def _apply_tape_flutter(audio: np.ndarray, sample_rate: int, depth: float = 0.0) -> np.ndarray:
    """
    Simulates analog tape wow & flutter (subtle capstan & motor drift)
    using fractional delay line with smooth linear interpolation.
    """
    if depth <= 1e-6 or audio.shape[1] < 1000:
        return audio

    num_samples = audio.shape[1]
    t = np.arange(num_samples) / sample_rate

    # Dual LFO: Slow capstan wow (~0.8 Hz) + fast motor flutter (~7.2 Hz)
    lfo = 0.7 * np.sin(2.0 * np.pi * 0.8 * t) + 0.3 * np.sin(2.0 * np.pi * 7.2 * t)
    delay_samples = 40.0 + (depth * sample_rate * lfo)

    indices = np.arange(num_samples) - delay_samples
    indices = np.clip(indices, 0, num_samples - 2)
    idx_floor = indices.astype(np.int64)
    frac = (indices - idx_floor).astype(np.float32)

    out = np.zeros_like(audio)
    for ch in range(audio.shape[0]):
        out[ch] = (1.0 - frac) * audio[ch, idx_floor] + frac * audio[ch, idx_floor + 1]

    return out.astype(np.float32)


def apply_sweetener(
    audio: np.ndarray,
    sample_rate: int,
    config: SweetenerConfig = None
) -> np.ndarray:
    """
    Run full psychoacoustic sweetening chain.
    """
    if config is None:
        config = SweetenerConfig()

    # Step 1: Fletcher-Munson Equal Loudness & Surgical EQ using Pedalboard C++ (L and R in parallel)
    eq_board = lambda: pedalboard.Pedalboard([
        pedalboard.HighpassFilter(cutoff_frequency_hz=config.sub_cut_hz),
        pedalboard.PeakFilter(
            cutoff_frequency_hz=config.bass_boost_hz,
            gain_db=config.bass_boost_db,
            q=1.0
        ),
        pedalboard.PeakFilter(
            cutoff_frequency_hz=config.de_mud_hz,
            gain_db=config.de_mud_db,
            q=1.2
        ),
        pedalboard.PeakFilter(
            cutoff_frequency_hz=config.harshness_cut_hz,
            gain_db=config.harshness_cut_db,
            q=1.1
        ),
        pedalboard.HighShelfFilter(
            cutoff_frequency_hz=config.air_sheen_hz,
            gain_db=config.air_sheen_db
        ),
    ])

    audio_eq = per_channel(lambda ch: eq_board()(ch, sample_rate), audio)

    # Step 2: Mid/Side Spatial Processing & Mono Low-End
    audio_spatial = _apply_mid_side_processing(
        audio_eq,
        sample_rate,
        mono_cutoff_hz=config.mono_bass_cutoff_hz,
        stereo_width=config.stereo_width
    )

    # Step 3: Harmonic Warmth Saturation
    audio_warmed = _apply_harmonic_warmth(audio_spatial, drive=config.warmth_drive)

    # Step 4: Analog Tape Wow & Flutter (if configured)
    if config.tape_flutter_depth > 1e-6:
        audio_warmed = _apply_tape_flutter(audio_warmed, sample_rate, depth=config.tape_flutter_depth)

    return audio_warmed
