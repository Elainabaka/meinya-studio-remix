"""
Sped Up & Jersey Bounce Preset
1.33x high-speed club energy, +4.0 semitones with vocal formant preservation,
differential transient punch on kicks/claps, sparkling sheen, and tight dynamic control.
"""

from typing import Optional

from .base import RemixPreset
from core.pipeline import RemixPipeline
from core.transient import TransientConfig
from core.sweetener import SweetenerConfig
from core.reverb_engine import ReverbConfig
from core.mastering import MasteringConfig
from core.deesser import DeEsserConfig


class SpedUpPreset(RemixPreset):
    @property
    def name(self) -> str:
        return "Sped Up & Jersey Bounce"

    @property
    def slug(self) -> str:
        return "sped_up"

    @property
    def description(self) -> str:
        return "1.33x speedup, +4 st pitch with formant preservation, snappy transient punch & club energy."

    def build_pipeline(
        self,
        speed: float = 1.33,
        pitch_semitones: Optional[float] = None,  # None = vinyl (pitch follows speed)
        preserve_formants: bool = True,
        reverb_amount: float = 0.08,
        **kwargs
    ) -> RemixPipeline:
        transient_cfg = TransientConfig(
            attack_gain=0.35,   # Extra transient snap on beats
            sustain_gain=-0.10  # Tight decay to prevent clutter at 1.33x speed
        )

        sweetener_cfg = SweetenerConfig(
            sub_cut_hz=35.0,
            bass_boost_hz=58.0,         # vinyl 1.33x lifts kick fundamentals into 55-65 Hz
            bass_boost_db=2.5,
            de_mud_hz=360.0,
            de_mud_db=-2.2,
            harshness_cut_hz=3600.0,
            harshness_cut_db=-1.5,
            air_sheen_hz=13000.0,
            air_sheen_db=2.2,           # pitch-up already brightens; more = sibilance
            mono_bass_cutoff_hz=115.0,
            stereo_width=1.25,
            warmth_drive=0.03
        )

        reverb_cfg = ReverbConfig(
            room_size=0.30,
            damping=0.65,
            wet_level=reverb_amount,
            dry_level=1.0,
            width=1.0,
            abbey_road_hp_hz=500.0,
            abbey_road_lp_hz=7200.0,
            ducking_amount=0.30
        )

        mastering_cfg = MasteringConfig(
            target_lufs=-9.0,
            true_peak_ceiling_db=-0.8,
            comp_threshold_db=-10.0,
            comp_ratio=2.0
        )

        return RemixPipeline(
            speed=speed,
            pitch_semitones=pitch_semitones or 0.0,
            vinyl_mode=pitch_semitones is None,
            preserve_formants=preserve_formants,
            transient_mode="crisp",
            transient_config=transient_cfg,
            sweetener_config=sweetener_cfg,
            reverb_config=reverb_cfg,
            mastering_config=mastering_cfg,
            deesser_config=DeEsserConfig(max_reduction_db=6.0)  # pitch-up pushes 's' into the harsh zone
        )
