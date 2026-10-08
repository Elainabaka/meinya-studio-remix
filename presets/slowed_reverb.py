"""
Slowed + Reverb Preset (Midnight Drift)
0.85x speed, -2.0 semitone pitch shift, Abbey Road filtered lush reverb,
tape warmth, wide stereo field, and punchy mono sub-bass.
"""

from typing import Optional

from .base import RemixPreset
from core.pipeline import RemixPipeline
from core.sweetener import SweetenerConfig
from core.reverb_engine import ReverbConfig
from core.mastering import MasteringConfig


class SlowedReverbPreset(RemixPreset):
    @property
    def name(self) -> str:
        return "Slowed + Reverb (Midnight Drift)"

    @property
    def slug(self) -> str:
        return "slowed_reverb"

    @property
    def description(self) -> str:
        return "0.85x speed, deep pitch, lush dynamic reverb, tape warmth & punchy sub."

    def build_pipeline(
        self,
        speed: float = 0.85,
        pitch_semitones: Optional[float] = None,  # None = vinyl (pitch follows speed)
        reverb_amount: float = 0.35,
        preserve_formants: bool = True,
        **kwargs
    ) -> RemixPipeline:
        sweetener_cfg = SweetenerConfig(
            sub_cut_hz=28.0,
            bass_boost_hz=70.0,
            bass_boost_db=3.2,
            de_mud_hz=340.0,
            de_mud_db=-2.5,
            harshness_cut_hz=3400.0,
            harshness_cut_db=-2.0,
            air_sheen_hz=12000.0,
            air_sheen_db=1.8,
            mono_bass_cutoff_hz=120.0,
            stereo_width=1.35,
            warmth_drive=0.06
        )

        reverb_cfg = ReverbConfig(
            room_size=0.82,
            damping=0.40,
            wet_level=reverb_amount,
            dry_level=1.0,
            width=1.0,
            abbey_road_hp_hz=420.0,
            abbey_road_lp_hz=6200.0,
            ducking_amount=0.38
        )

        mastering_cfg = MasteringConfig(
            target_lufs=-11.0,
            true_peak_ceiling_db=-1.0,
            comp_threshold_db=-13.0,
            comp_ratio=1.6
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
