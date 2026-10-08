"""
Phonk & Bass Overdrive Preset
Deep 65Hz sub-bass boost, aggressive dynamic compression,
surgical de-mudding, and strict mono sub-bass focus.
"""

from .base import RemixPreset
from core.pipeline import RemixPipeline
from core.transient import TransientConfig
from core.sweetener import SweetenerConfig
from core.reverb_engine import ReverbConfig
from core.mastering import MasteringConfig


class BassBoostPreset(RemixPreset):
    @property
    def name(self) -> str:
        return "Drift Phonk & Bass Overdrive"

    @property
    def slug(self) -> str:
        return "bass_boost"

    @property
    def description(self) -> str:
        return "Heavy 65Hz sub punch, pumping dynamics, mono low-end focus, wide stereo highs."

    def build_pipeline(
        self,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        bass_boost_db: float = 5.5,
        **kwargs
    ) -> RemixPipeline:
        transient_cfg = TransientConfig(
            attack_gain=0.40,   # Massive sub-kick attack
            sustain_gain=0.15   # Extended 808 sub sustain body
        )

        sweetener_cfg = SweetenerConfig(
            sub_cut_hz=28.0,
            bass_boost_hz=65.0,
            bass_boost_db=bass_boost_db,
            de_mud_hz=320.0,
            de_mud_db=-3.0,
            harshness_cut_hz=3500.0,
            harshness_cut_db=-1.5,
            air_sheen_hz=12000.0,
            air_sheen_db=2.5,
            mono_bass_cutoff_hz=130.0,
            stereo_width=1.40,
            warmth_drive=0.08
        )

        mastering_cfg = MasteringConfig(
            target_lufs=-9.0,
            true_peak_ceiling_db=-0.8,
            comp_threshold_db=-9.5,
            comp_ratio=2.2,
            comp_attack_ms=15.0,
            comp_release_ms=80.0
        )

        return RemixPipeline(
            speed=speed,
            pitch_semitones=pitch_semitones,
            transient_config=transient_cfg,
            sweetener_config=sweetener_cfg,
            mastering_config=mastering_cfg
        )
