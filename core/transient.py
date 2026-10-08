"""
Differential Envelope Transient Shaper Module
SPL Transient Designer style differential envelope follower:
Separates attack transients (fast envelope, ~2ms) from steady-state body (slow envelope, ~25ms)
to sharpen percussion punch or soften aggressive hits.
"""

from dataclasses import dataclass
import numpy as np
from scipy import signal


@dataclass
class TransientConfig:
    attack_gain: float = 0.0   # -1.0 (softer attack) to +1.0 (punchy snappy attack)
    sustain_gain: float = 0.0  # -1.0 (tighter decay) to +1.0 (extended sustained body)
    fast_ms: float = 2.0       # Attack detection window (~2ms)
    slow_ms: float = 25.0      # Sustain detection window (~25ms)


def apply_transient_shaping(
    audio: np.ndarray,
    sample_rate: int,
    config: TransientConfig = None
) -> np.ndarray:
    """
    Applies differential envelope transient shaping.
    Audio shape: (channels, samples), float32.
    """
    if config is None:
        config = TransientConfig()

    if abs(config.attack_gain) < 0.01 and abs(config.sustain_gain) < 0.01:
        return audio

    num_samples = audio.shape[1]
    if num_samples < 200:
        return audio

    # Compute absolute rectified mono signal
    rectified = np.mean(np.abs(audio), axis=0)

    # Convert ms time constants to 1-pole IIR smoothing coefficients
    # alpha = exp(-1 / (time_const * sample_rate))
    alpha_fast = float(np.exp(-1.0 / (config.fast_ms * 0.001 * sample_rate)))
    alpha_slow = float(np.exp(-1.0 / (config.slow_ms * 0.001 * sample_rate)))

    # Compute envelopes using 1-pole lowpass filter: y[n] = (1 - a)*x[n] + a*y[n-1]
    b_fast = [1.0 - alpha_fast]
    a_fast = [1.0, -alpha_fast]
    env_fast = signal.lfilter(b_fast, a_fast, rectified)

    b_slow = [1.0 - alpha_slow]
    a_slow = [1.0, -alpha_slow]
    env_slow = signal.lfilter(b_slow, a_slow, rectified)

    # Differential envelope represents instantaneous transient attack
    diff = env_fast - env_slow

    # Normalize differential
    ref_level = float(np.percentile(np.abs(diff[diff > 1e-4]), 95)) if np.any(diff > 1e-4) else float(np.max(np.abs(diff)) + 1e-6)
    diff_norm = diff / (ref_level + 1e-6)

    # Construct shaping gain curve
    # Positive diff = attack transient; negative diff or slow env = sustain
    transient_gain = 1.0 + (config.attack_gain * np.clip(diff_norm, 0.0, 2.5))
    
    if abs(config.sustain_gain) > 0.01:
        sustain_ref = float(np.percentile(env_slow[env_slow > 1e-4], 90)) if np.any(env_slow > 1e-4) else float(np.max(env_slow) + 1e-6)
        sustain_norm = np.clip(env_slow / (sustain_ref + 1e-6), 0.0, 2.0)
        sustain_mod = 1.0 + (config.sustain_gain * sustain_norm * (1.0 - np.clip(diff_norm, 0.0, 1.0)))
        transient_gain = transient_gain * sustain_mod

    # Bound gain between 0.2 (-14dB) and 2.5 (+8dB) for stability
    transient_gain = np.clip(transient_gain, 0.2, 2.5).astype(np.float32)

    shaped = audio * transient_gain[np.newaxis, :]
    return shaped.astype(np.float32)
