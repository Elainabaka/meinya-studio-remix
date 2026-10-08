"""
Nightcore Deluxe Preset
1.25x tempo, +3.0 semitones pitch with vocal formant preservation,
snappy transient punch, sparkling air sheen, and club-ready loudness.
"""

from typing import Optional

from .base import RemixPreset
from core.pipeline import RemixPipeline
from core.transient import TransientConfig
from core.sweetener import SweetenerConfig
from core.reverb_engine import ReverbConfig
from core.mastering import MasteringConfig
from core.deesser import DeEsserConfig


class NightcorePreset(RemixPreset):
    @property
    def name(self) -> str:
        return "Nightcore Deluxe (High-Fidelity Speedup)"

    @property
    def slug(self) -> str:
        return "nightcore"

    @property
    def description(self) -> str:
        return "1.25x tempo, +3 semitones with formant preservation, snappy transients & sparkling sheen."

    def build_pipeline(
        self,
        speed: float = 1.25,
        pitch_semitones: Optional[float] = None,  # None = vinyl (pitch follows speed)
        preserve_formants: bool = True,
        reverb_amount: float = 0.12,
        **kwargs
    ) -> RemixPipeline:
        transient_cfg = TransientConfig(
            attack_gain=0.30,   # Snappy transient snap on kicks and claps
            sustain_gain=-0.05
        )

        sweetener_cfg = SweetenerConfig(
            sub_cut_hz=32.0,
            bass_boost_hz=58.0,         # vinyl 1.25x lifts kick fundamentals ~40->50-60 Hz
            bass_boost_db=2.8,
            de_mud_hz=360.0,
            de_mud_db=-2.0,
            harshness_cut_hz=3600.0,
            harshness_cut_db=-1.2,
            air_sheen_hz=12500.0,
            air_sheen_db=2.0,           # pitch-up already brightens; more = sibilance
            mono_bass_cutoff_hz=110.0,
            stereo_width=1.20,
            warmth_drive=0.03
        )

        reverb_cfg = ReverbConfig(
            room_size=0.35,
            damping=0.60,
            wet_level=reverb_amount,
            dry_level=1.0,
            width=1.0,
            abbey_road_hp_hz=500.0,
            abbey_road_lp_hz=7000.0,
            ducking_amount=0.25
        )

        mastering_cfg = MasteringConfig(
            target_lufs=-9.5,
            true_peak_ceiling_db=-0.8,
            comp_threshold_db=-11.0,
            comp_ratio=1.8
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
