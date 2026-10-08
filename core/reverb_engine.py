"""
Abbey Road Reverb Engine with Dynamic Sidechain Ducking
Filters out low-end mud (<400Hz) and harsh top sizzle (>6500Hz) on the reverb send,
and automatically ducks the wet tail when sharp transients hit to keep vocals and drums crystal clear.
"""

from dataclasses import dataclass
import numpy as np
from scipy import signal
import pedalboard

from .channels import per_channel


@dataclass
class ReverbConfig:
    room_size: float = 0.75         # Spacious cathedral/hall feel (0.0 to 1.0)
    damping: float = 0.45           # Natural high frequency damping
    wet_level: float = 0.32         # Wet reverb blend ratio
    dry_level: float = 1.0          # Original signal level
    width: float = 1.0              # Reverb stereo spread (clamped to 0.0-1.0)
    abbey_road_hp_hz: float = 380.0 # Highpass send: keeps kick & bass out of reverb
    abbey_road_lp_hz: float = 6800.0# Lowpass send: prevents harsh sibilance
    ducking_amount: float = 0.35    # Dynamic ducking depth (0.0 = off, 0.4 = -4.5dB duck)
    pre_delay_ms: float = 25.0      # Gap before the tail starts: keeps vocal consonants intelligible
    tail_sec: float = -1.0          # Silence appended so the last tail rings out (<0 = auto from room_size)


def apply_abbey_road_reverb(
    audio: np.ndarray,
    sample_rate: int,
    config: ReverbConfig = None,
    send: np.ndarray = None
) -> np.ndarray:
    """
    Applies studio-grade Abbey Road filtered reverb with dynamic envelope ducking.
    `send`: what feeds the reverb (default: the dry audio itself; stem mode passes vocals + other).
    Ducking is always keyed on the dry audio. Output is longer than the input by the reverb tail.
    """
    if config is None:
        config = ReverbConfig()

    if config.wet_level <= 0.001:
        return audio * config.dry_level

    # Step 0: Append room for the final tail (otherwise it is cut off at the last sample)
    tail_sec = config.tail_sec if config.tail_sec >= 0 else 1.0 + 4.0 * config.room_size
    tail = int(tail_sec * sample_rate)
    send = audio if send is None else send[:, :audio.shape[1]]
    audio = np.pad(audio, ((0, 0), (0, tail)))
    send = np.pad(send, ((0, 0), (0, audio.shape[1] - send.shape[1]))).astype(np.float32)

    # Step 1: Filter the send signal using Abbey Road bandpass
    send_filter = lambda: pedalboard.Pedalboard([
        pedalboard.HighpassFilter(cutoff_frequency_hz=config.abbey_road_hp_hz),
        pedalboard.LowpassFilter(cutoff_frequency_hz=config.abbey_road_lp_hz)
    ])
    send_audio = per_channel(lambda ch: send_filter()(ch, sample_rate), send)

    # Step 1.5: Pre-delay on the send only (dry stays in time)
    pre = int(config.pre_delay_ms * 0.001 * sample_rate)
    if pre > 0:
        send_audio = np.pad(send_audio, ((0, 0), (pre, 0)))[:, :audio.shape[1]]

    # Step 2: Generate 100% wet reverb from the filtered send
    reverb_width = float(np.clip(config.width, 0.0, 1.0))
    reverb_processor = pedalboard.Pedalboard([
        pedalboard.Reverb(
            room_size=config.room_size,
            damping=config.damping,
            wet_level=1.0,
            dry_level=0.0,
            width=reverb_width
        )
    ])
    wet_audio = reverb_processor(send_audio, sample_rate)

    # Step 3: Dynamic Adaptive Ducking
    # When dry signal is active, compress the reverb tail to preserve transient punch
    if config.ducking_amount > 0.01:
        dry_mono = np.mean(np.abs(audio), axis=0)
        # 8 Hz lowpass filter to extract smooth volume envelope
        sos = signal.butter(2, 8.0, btype='lowpass', fs=sample_rate, output='sos')
        env = signal.sosfiltfilt(sos, dry_mono)
        
        # Adaptive 95th percentile follower prevents a single spike from muting ducking sensitivity
        active_mask = env > 1e-4
        ref_val = float(np.percentile(env[active_mask], 95)) if np.any(active_mask) else float(np.max(env))
        
        if ref_val > 1e-6:
            env_norm = np.clip(env / ref_val, 0.0, 1.5)
            # Smooth ducking gain curve
            ducking_gain = 1.0 - (config.ducking_amount * (env_norm / 1.5))
            ducking_gain = np.clip(ducking_gain, 0.15, 1.0)
            wet_audio = wet_audio * ducking_gain[np.newaxis, :]

    # Step 4: Mix Dry + Wet, fade the very end of the tail to avoid a click
    output = (audio * config.dry_level) + (wet_audio * config.wet_level)
    fade = min(int(0.5 * sample_rate), tail)
    if fade > 0:
        output[:, -fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    return output.astype(np.float32)
