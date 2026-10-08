"""
Time-Stretch and Pitch-Shift Processing Module
Powered by Spotify Pedalboard C++ Rubber Band engine with high-fidelity transient detection
and formant preservation.
"""

from fractions import Fraction

import numpy as np
import pedalboard
from scipy import signal
from typing import Optional

from .channels import per_channel


def varispeed(audio: np.ndarray, speed: float) -> np.ndarray:
    """
    Classic turntable/tape varispeed: tempo and pitch move together (pitch = 12*log2(speed) st).
    Pure band-limited resampling -> no phase-vocoder artifacts at all; this is how the
    original Nightcore and Slowed + Reverb genres were made.
    """
    ratio = Fraction(1.0 / speed).limit_denominator(1000)
    out = per_channel(lambda ch: signal.resample_poly(
        ch, ratio.numerator, ratio.denominator, axis=1, window=("kaiser", 8.6)
    ), audio)
    return np.ascontiguousarray(out, dtype=np.float32)


def time_pitch_process(
    audio: np.ndarray,
    sample_rate: int,
    speed: float = 1.0,
    pitch_semitones: float = 0.0,
    preserve_formants: bool = True,
    vinyl_mode: bool = False,
    transient_mode: str = "crisp"
) -> np.ndarray:
    """
    Adjust tempo and pitch of stereo audio.
    
    Args:
        audio: np.ndarray with shape (channels, samples), float32.
        sample_rate: Sample rate (e.g. 44100 or 48000).
        speed: Speed multiplier (1.0 = unchanged, 1.25 = faster, 0.85 = slower).
        pitch_semitones: Pitch offset in semitones (0.0 = unchanged, +2 = higher, -2 = lower).
        preserve_formants: Keeps vocal body natural when shifting pitch.
        vinyl_mode: If True, pitch follows speed like a turntable (true resampling, pitch_semitones ignored).
        transient_mode: 'crisp', 'mixed', or 'smooth'. NOTE: only the R2 engine (high_quality=False)
            uses it; the R3 engine used here ignores it (verified: identical output).

    Returns:
        Processed audio numpy array (channels, new_samples).
    """
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)

    # In vinyl mode, pitch is physically locked to speed: resample instead of phase vocoder
    if vinyl_mode and speed > 0:
        if abs(speed - 1.0) < 1e-4:
            return audio.copy()
        return varispeed(audio, speed)

    # Fast path if no change requested
    if abs(speed - 1.0) < 1e-4 and abs(pitch_semitones) < 1e-4:
        return audio.copy()

    # Audio format check for pedalboard
    audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)

    try:
        stretched = pedalboard.time_stretch(
            audio_f32,
            float(sample_rate),
            stretch_factor=float(speed),
            pitch_shift_in_semitones=float(pitch_semitones),
            high_quality=True,
            transient_mode=transient_mode,
            preserve_formants=preserve_formants,
            retain_phase_continuity=True
        )
        return stretched
    except Exception as e:
        # Graceful fallback or re-raise
        raise RuntimeError(f"DSP time_pitch_process failed: {e}")
