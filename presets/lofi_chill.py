"""
Lo-Fi Nostalgia Preset
0.92x speed, vintage analog tape warmth, cozy mellow roll-off,
subtle room reverb, and intimate dynamic range.
"""

from typing import Optional

from .base import RemixPreset
from core.pipeline import RemixPipeline
from core.sweetener import SweetenerConfig
from core.reverb_engine import ReverbConfig
from core.mastering import MasteringConfig


class LoFiChillPreset(RemixPreset):
    @property
    def name(self) -> str:
        return "Lo-Fi Nostalgia (Warm & Cozy Tape)"

    @property
    def slug(self) -> str:
        return "lofi_chill"

    @property
    def description(self) -> str:
        return "0.92x speed, vintage tape saturation, mellow high roll-off, intimate cozy room reverb."

    def build_pipeline(
        self,
        speed: float = 0.92,
        pitch_semitones: Optional[float] = None,  # None = vinyl (pitch follows speed)
        reverb_amount: float = 0.25,
        **kwargs
    ) -> RemixPipeline:
        sweetener_cfg = SweetenerConfig(
            sub_cut_hz=40.0,
            bass_boost_hz=120.0,
            bass_boost_db=2.5,
            de_mud_hz=380.0,
            de_mud_db=-1.5,
            harshness_cut_hz=3800.0,
            harshness_cut_db=-2.5,
            air_sheen_hz=10000.0,
            air_sheen_db=-2.0,          # Mellow tape high roll-off
            mono_bass_cutoff_hz=110.0,
            stereo_width=1.10,
            warmth_drive=0.08,          # Vintage tape saturation
            tape_flutter_depth=0.0004   # Subtle analog tape reel wobble
        )

        reverb_cfg = ReverbConfig(
            room_size=0.50,
            damping=0.70,
            wet_level=reverb_amount,
            dry_level=1.0,
            width=0.9,
            abbey_road_hp_hz=350.0,
            abbey_road_lp_hz=5500.0,
            ducking_amount=0.20
        )

        mastering_cfg = MasteringConfig(
            target_lufs=-13.0,
            true_peak_ceiling_db=-1.2,
            comp_threshold_db=-14.0,
            comp_ratio=1.4
        )

        return RemixPipeline(
            speed=speed,
            pitch_semitones=pitch_semitones or 0.0,
            vinyl_mode=pitch_semitones is None,
            preserve_formants=True,
            sweetener_config=sweetener_cfg,
            reverb_config=reverb_cfg,
            mastering_config=mastering_cfg
        )
