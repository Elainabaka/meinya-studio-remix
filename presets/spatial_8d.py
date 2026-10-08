"""
8D Binaural Surround Preset
Simulates 360-degree rotating surround sound around the listener's head
using constant-power pan modulation, interaural phase depth, and ambient reflection.
"""

from typing import Optional, Callable
import numpy as np
from scipy import signal
from .base import RemixPreset
from core.pipeline import RemixPipeline
from core.sweetener import SweetenerConfig
from core.reverb_engine import ReverbConfig
from core.mastering import MasteringConfig, master_audio


def apply_8d_rotation(
    audio: np.ndarray,
    sample_rate: int,
    rotation_speed_hz: float = 0.15,
    bass_cutoff_hz: float = 150.0,
    max_itd_sec: float = 0.00063,
    ambience: float = 0.4
) -> np.ndarray:
    """
    Orbits the mix around the listener's head (360 deg every ~6.6 s at 0.15 Hz):
    - constant-power pan (sin/cos law) for the level difference between ears (ILD),
    - interaural time delay on the far ear (ITD, Woodworth max ~0.63 ms),
    - darker tone while the source is behind (pinna shadow) -> front/back cue,
    - sub-bass below bass_cutoff_hz stays centered: it carries almost no localization cue
      and one-sided bass on headphones is fatiguing,
    - part of the original Side (room/reverb) stays static so the space does not collapse.
    """
    if audio.shape[0] < 2:
        audio = np.stack([audio[0], audio[0]], axis=0)

    num_samples = audio.shape[1]
    t = np.arange(num_samples) / sample_rate

    # Zero-phase band split with perfect reconstruction (high = full - low)
    sos = signal.butter(4, bass_cutoff_hz, btype="lowpass", fs=sample_rate, output="sos")
    low = signal.sosfiltfilt(sos, audio, axis=1)
    high = audio - low
    src = 0.5 * (high[0] + high[1])
    side = 0.5 * (high[0] - high[1])

    # Azimuth: 0 = front, pi/2 = right, pi = behind, 3pi/2 = left
    azimuth = 2.0 * np.pi * rotation_speed_hz * t
    pan = np.sin(azimuth)                      # -1 left .. +1 right
    behind = 0.5 * (1.0 - np.cos(azimuth))     # 0 front .. 1 behind

    # Back-darkening: blend toward a 4 kHz lowpassed copy while behind
    sos_back = signal.butter(2, 4000.0, btype="lowpass", fs=sample_rate, output="sos")
    dark = signal.sosfiltfilt(sos_back, src)
    src = src + 0.45 * behind * (dark - src)

    # ITD: fractional delay on the far ear only
    n = np.arange(num_samples, dtype=np.float64)
    delay_l = np.maximum(pan, 0.0) * max_itd_sec * sample_rate
    delay_r = np.maximum(-pan, 0.0) * max_itd_sec * sample_rate
    src_l = np.interp(n - delay_l, n, src)
    src_r = np.interp(n - delay_r, n, src)

    # Constant-power pan (-3 dB each ear at center)
    theta = (pan + 1.0) * (np.pi / 4.0)
    out_l = low[0] + src_l * np.cos(theta) + ambience * side
    out_r = low[1] + src_r * np.sin(theta) - ambience * side

    return np.stack([out_l, out_r], axis=0).astype(np.float32)


class Spatial8DPreset(RemixPreset):
    @property
    def name(self) -> str:
        return "8D Binaural Surround (Orbiting 360°)"

    @property
    def slug(self) -> str:
        return "spatial_8d"

    @property
    def description(self) -> str:
        return "Smooth 360° orbital surround rotation with binaural panning and spacious reflections."

    def build_pipeline(
        self,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        reverb_amount: float = 0.28,
        **kwargs
    ) -> RemixPipeline:
        sweetener_cfg = SweetenerConfig(
            sub_cut_hz=28.0,
            bass_boost_hz=75.0,
            bass_boost_db=2.2,
            de_mud_hz=350.0,
            de_mud_db=-2.0,
            harshness_cut_hz=3500.0,
            harshness_cut_db=-1.5,
            air_sheen_hz=12000.0,
            air_sheen_db=2.0,
            mono_bass_cutoff_hz=100.0,
            stereo_width=1.20,
            warmth_drive=0.03
        )

        reverb_cfg = ReverbConfig(
            room_size=0.70,
            damping=0.50,
            wet_level=reverb_amount,
            dry_level=1.0,
            width=1.0,
            abbey_road_hp_hz=350.0,
            abbey_road_lp_hz=6500.0,
            ducking_amount=0.30
        )

        mastering_cfg = MasteringConfig(
            target_lufs=-11.0,
            true_peak_ceiling_db=-1.0
        )

        return RemixPipeline(
            speed=speed,
            pitch_semitones=pitch_semitones,
            sweetener_config=sweetener_cfg,
            reverb_config=reverb_cfg,
            mastering_config=mastering_cfg
        )

    def apply(
        self,
        audio: np.ndarray,
        sample_rate: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        knobs: Optional[dict] = None,
        stems: Optional[dict] = None,
        rotation_speed_hz: float = 0.15,
        **kwargs
    ) -> np.ndarray:
        pipeline = self.build(knobs, **kwargs)
        processed = pipeline.process(audio, sample_rate, progress_callback=progress_callback, stems=stems)

        if progress_callback:
            progress_callback("Applying 8D Spatial Orbit Modulation...", 0.85)
        orbit_audio = apply_8d_rotation(processed, sample_rate, rotation_speed_hz=rotation_speed_hz)

        if progress_callback:
            progress_callback("Finalizing 8D Limiter Protection...", 0.95)
        final_audio = master_audio(orbit_audio, sample_rate, config=pipeline.mastering_config)

        if progress_callback:
            progress_callback("8D Remix Complete!", 1.0)
        return final_audio
