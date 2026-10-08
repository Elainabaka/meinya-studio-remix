"""
Studio Mastering Module
Implements EBU R128 Loudness Normalization, gentle master bus compression,
and ITU-R BS.1770 compliant lookahead True-Peak Brickwall Limiting.
"""

from dataclasses import dataclass
import numpy as np
from .loudness import integrated_loudness
import pedalboard

from .channels import per_channel


@dataclass
class MasteringConfig:
    target_lufs: float = -10.5          # Standard loud & punchy remix level (-14.0 for hi-fi)
    true_peak_ceiling_db: float = -1.0  # Headroom protection against codec clipping
    apply_glue_compression: bool = True # Smooth master bus glue
    comp_threshold_db: float = -12.0
    comp_ratio: float = 1.6
    comp_attack_ms: float = 25.0
    comp_release_ms: float = 120.0
    enable_brickwall: bool = True


def master_audio(
    audio: np.ndarray,
    sample_rate: int,
    config: MasteringConfig = None
) -> np.ndarray:
    """
    Applies master bus glue compression, loudness normalization, and true-peak limiting.
    Audio input shape: (channels, samples).
    """
    if config is None:
        config = MasteringConfig()

    out_audio = audio.copy()

    # Step 1: Gentle Glue Compression (binds transients together smoothly)
    if config.apply_glue_compression:
        # Per-channel detector in JUCE's compressor: L and R in parallel give the same samples
        glue_comp = lambda: pedalboard.Pedalboard([
            pedalboard.Compressor(
                threshold_db=config.comp_threshold_db,
                ratio=config.comp_ratio,
                attack_ms=config.comp_attack_ms,
                release_ms=config.comp_release_ms
            )
        ])
        out_audio = per_channel(lambda ch: glue_comp()(ch, sample_rate), out_audio)

    # Step 2: Measure EBU R128 Integrated Loudness (same value as pyloudnorm, core/loudness.py)
    try:
        current_lufs = integrated_loudness(out_audio, sample_rate)
        
        # Guard against silence or invalid LUFS
        if not np.isnan(current_lufs) and not np.isinf(current_lufs) and current_lufs > -70.0:
            gain_db = config.target_lufs - current_lufs
            # Cap maximum automatic boost at +12dB for safety
            gain_db = min(gain_db, 12.0)
            linear_gain = 10.0 ** (gain_db / 20.0)
            out_audio = out_audio * linear_gain
    except Exception:
        # Fallback to RMS normalization if meter encounters edge case
        rms = np.sqrt(np.mean(out_audio ** 2))
        if rms > 1e-5:
            target_rms = 0.18
            out_audio = out_audio * (target_rms / rms)

    # Step 3: ITU-R BS.1770 4x Oversampled True-Peak Brickwall Limiting
    if config.enable_brickwall:
        # Stereo-linked (one gain for L and R): never split per channel (core/channels.py)
        limiter = pedalboard.Pedalboard([
            pedalboard.BrickwallLimiter(
                ceiling_db=config.true_peak_ceiling_db,
                true_peak=True
            )
        ])
        out_audio = limiter(out_audio, sample_rate)

    return np.ascontiguousarray(out_audio, dtype=np.float32)
