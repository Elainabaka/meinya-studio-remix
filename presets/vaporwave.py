"""
Vaporwave & Mallsoft Aesthetic Preset
0.75x extreme slow, -3.5 semitones pitch shift, analog tape wow & flutter,
vintage modulated space, and warm dreamy saturation.
"""

from typing import Optional

from .base import RemixPreset
from core.pipeline import RemixPipeline
from core.sweetener import SweetenerConfig
from core.reverb_engine import ReverbConfig
from core.mastering import MasteringConfig


class VaporwavePreset(RemixPreset):
    @property
    def name(self) -> str:
        return "Vaporwave & Mallsoft Aesthetic"

    @property
    def slug(self) -> str:
        return "vaporwave"

    @property
    def description(self) -> str:
        return "0.75x slow, -3.5 st pitch, analog tape wow & flutter, lush dreamy hall reverb."

    def build_pipeline(
        self,
        speed: float = 0.75,
        pitch_semitones: Optional[float] = None,  # None = vinyl (pitch follows speed)
        preserve_formants: bool = True,
        reverb_amount: float = 0.42,
        **kwargs
    ) -> RemixPipeline:
        sweetener_cfg = SweetenerConfig(
            sub_cut_hz=26.0,
            bass_boost_hz=68.0,
            bass_boost_db=3.5,
            de_mud_hz=330.0,
            de_mud_db=-2.8,
            harshness_cut_hz=3200.0,
            harshness_cut_db=-2.5,
            air_sheen_hz=11000.0,
            air_sheen_db=-1.0,           # Cozy vintage high roll-off
            mono_bass_cutoff_hz=125.0,
            stereo_width=1.40,
            warmth_drive=0.07,
            tape_flutter_depth=0.0006   # Authentic tape reel wobble
        )

        reverb_cfg = ReverbConfig(
            room_size=0.88,              # Vast empty shopping mall reverb
            damping=0.35,
            wet_level=reverb_amount,
            dry_level=0.95,
            width=1.0,
            abbey_road_hp_hz=360.0,
            abbey_road_lp_hz=5800.0,
            ducking_amount=0.35
        )

        mastering_cfg = MasteringConfig(
            target_lufs=-12.5,
            true_peak_ceiling_db=-1.2,
            comp_threshold_db=-14.0,
            comp_ratio=1.5
        )

        return RemixPipeline(
            speed=speed,
            pitch_semitones=pitch_semitones or 0.0,
            vinyl_mode=pitch_semitones is None,
            preserve_formants=preserve_formants,
            transient_mode="crisp",
            sweetener_config=sweetener_cfg,
            reverb_config=reverb_cfg,
            mastering_config=mastering_cfg
        )
