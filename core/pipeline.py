"""
Remix Pipeline: Orchestrates audio stages
(Gain staging -> Time/Pitch -> Transient -> De-ess -> Sweetener -> Reverb -> Mastering)
with progress tracking and optional stem-aware steering.
"""

from typing import Optional, Callable, Dict
import numpy as np
from .loudness import integrated_loudness
from .time_pitch import time_pitch_process
from .transient import apply_transient_shaping, TransientConfig
from .deesser import apply_deesser, DeEsserConfig
from .sweetener import apply_sweetener, SweetenerConfig
from .reverb_engine import apply_abbey_road_reverb, ReverbConfig
from .mastering import master_audio, MasteringConfig


# Gain staging reference: analog-modelled stages are calibrated around -18 dBFS RMS (0 VU).
STAGING_LUFS = -18.0
STAGING_PEAK_DB = -6.0

# Stem mode: how much of each stem feeds the reverb (drums & bass stay dry -> punch and clarity)
STEM_REVERB_SEND = {"vocals": 1.0, "other": 0.8, "drums": 0.15}


def staging_gain(audio: np.ndarray, sample_rate: int) -> float:
    """
    Linear gain that brings the source to a fixed internal level before any non-linear stage.
    Without this, a hot -5 LUFS master hits the tanh warmth / glue compressor ~13 dB harder
    than a -18 LUFS track, so distortion depended on the source instead of the preset.
    """
    peak_db = 20.0 * np.log10(float(np.max(np.abs(audio))) + 1e-12)
    if peak_db < -90.0:
        return 1.0
    gain_db = STAGING_PEAK_DB - peak_db
    try:
        lufs = integrated_loudness(audio, sample_rate)
        if np.isfinite(lufs) and lufs > -70.0:
            gain_db = min(STAGING_LUFS - lufs, gain_db)
    except ValueError:
        pass  # clip shorter than one 400 ms gating block: peak rule only
    return float(10.0 ** (gain_db / 20.0))


def stage_input_gain(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return (audio * np.float32(staging_gain(audio, sample_rate))).astype(np.float32)


def _fit(audio: np.ndarray, num_samples: int) -> np.ndarray:
    if audio.shape[1] >= num_samples:
        return audio[:, :num_samples]
    return np.pad(audio, ((0, 0), (0, num_samples - audio.shape[1])))


class RemixPipeline:
    """
    High-Fidelity Audio Remixing Pipeline.
    Modular, chainable, and observable.
    """

    def __init__(
        self,
        speed: float = 1.0,
        pitch_semitones: float = 0.0,
        preserve_formants: bool = True,
        vinyl_mode: bool = False,
        transient_mode: str = "crisp",
        transient_config: Optional[TransientConfig] = None,
        sweetener_config: Optional[SweetenerConfig] = None,
        reverb_config: Optional[ReverbConfig] = None,
        mastering_config: Optional[MasteringConfig] = None,
        deesser_config: Optional[DeEsserConfig] = None,
    ):
        self.speed = speed
        self.pitch_semitones = pitch_semitones
        self.preserve_formants = preserve_formants
        self.vinyl_mode = vinyl_mode
        self.transient_mode = transient_mode
        self.transient_config = transient_config
        self.sweetener_config = sweetener_config or SweetenerConfig()
        self.reverb_config = reverb_config
        self.mastering_config = mastering_config or MasteringConfig()
        self.deesser_config = deesser_config

    def get_knobs(self) -> dict:
        """User-facing knobs shared by every preset. pitch=None means vinyl (pitch follows speed)."""
        return {
            "speed": self.speed,
            "pitch": None if self.vinyl_mode else self.pitch_semitones,
            "reverb": self.reverb_config.wet_level if self.reverb_config else 0.0,
            "bass_db": self.sweetener_config.bass_boost_db,
            "air_db": self.sweetener_config.air_sheen_db,
            "punch": self.transient_config.attack_gain if self.transient_config else 0.0,
            "width": self.sweetener_config.stereo_width,
            "lufs": self.mastering_config.target_lufs,
            "deess_db": self.deesser_config.max_reduction_db if self.deesser_config else 0.0,
        }

    def set_knobs(self, **knobs) -> "RemixPipeline":
        """Override knobs (absolute values) on top of a preset's pipeline. Unknown keys are ignored."""
        limits = {"speed": (0.5, 1.6), "pitch": (-12, 12), "reverb": (0, 0.8), "bass_db": (-6, 10),
                  "air_db": (-6, 6), "punch": (-0.5, 0.8), "width": (0.6, 1.8), "lufs": (-16, -7),
                  "deess_db": (0, 12)}
        values = {**self.get_knobs(), **knobs}
        for key, (lo, hi) in limits.items():
            if values.get(key) is not None:
                value = float(values[key])
                if isinstance(values[key], bool) or not np.isfinite(value) or not lo <= value <= hi:
                    raise ValueError(f"{key} must be finite and between {lo} and {hi}")
        if knobs.get("speed") is not None:
            self.speed = float(knobs["speed"])
        if "pitch" in knobs:
            self.vinyl_mode = knobs["pitch"] is None
            self.pitch_semitones = 0.0 if knobs["pitch"] is None else float(knobs["pitch"])
        if knobs.get("reverb") is not None:
            if self.reverb_config is None:
                self.reverb_config = ReverbConfig(wet_level=0.0)
            self.reverb_config.wet_level = float(knobs["reverb"])
        if knobs.get("bass_db") is not None:
            self.sweetener_config.bass_boost_db = float(knobs["bass_db"])
        if knobs.get("air_db") is not None:
            self.sweetener_config.air_sheen_db = float(knobs["air_db"])
        if knobs.get("punch") is not None:
            if self.transient_config is None:
                self.transient_config = TransientConfig()
            self.transient_config.attack_gain = float(knobs["punch"])
        if knobs.get("width") is not None:
            self.sweetener_config.stereo_width = float(knobs["width"])
        if knobs.get("lufs") is not None:
            self.mastering_config.target_lufs = float(knobs["lufs"])
        if knobs.get("deess_db") is not None:
            if self.deesser_config is None:
                self.deesser_config = DeEsserConfig(max_reduction_db=0.0)
            self.deesser_config.max_reduction_db = float(knobs["deess_db"])
        return self

    def _time_pitch(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        return time_pitch_process(
            audio,
            sample_rate,
            speed=self.speed,
            pitch_semitones=self.pitch_semitones,
            preserve_formants=self.preserve_formants,
            vinyl_mode=self.vinyl_mode,
            transient_mode=self.transient_mode
        )

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        stems: Optional[Dict[str, np.ndarray]] = None
    ) -> np.ndarray:
        """
        Executes the entire remix pipeline.

        Args:
            audio: np.ndarray shape (channels, samples).
            sample_rate: int sample rate.
            progress_callback: optional callback(stage_name, fraction).
            stems: optional {'vocals','drums','other',...} of the same mix (core.stems). They only
                steer processing: reverb send (vocals/other), drum-only transient shaping, vocal-only
                de-essing. The dry path stays `audio` + per-stem differences, so separation
                artifacts never reach the output directly.
        """
        self.set_knobs()  # constructor/CLI overrides use the same bounds as the UI

        def update(msg: str, frac: float):
            if progress_callback:
                progress_callback(msg, frac)

        transient_on = self.transient_config is not None and (
            abs(self.transient_config.attack_gain) > 0.01 or abs(self.transient_config.sustain_gain) > 0.01)
        deess_on = self.deesser_config is not None and self.deesser_config.max_reduction_db > 0.01
        reverb_on = self.reverb_config is not None and self.reverb_config.wet_level > 0.001

        # Stage 0: Gain staging so every non-linear stage sees the same level for any source
        update("Gain Staging Input...", 0.05)
        gain = np.float32(staging_gain(audio, sample_rate))
        staged = (audio * gain).astype(np.float32)

        # Stage 1: Time Stretching & Pitch Shifting (stems get the identical transform)
        update("Applying Time & Pitch Transformation...", 0.20)
        processed = self._time_pitch(staged, sample_rate)
        steer = {}
        if stems:
            needed = set(STEM_REVERB_SEND) if reverb_on else set()
            if transient_on:
                needed.add("drums")
            if deess_on:
                needed.add("vocals")
            for name in sorted(needed & set(stems)):
                steer[name] = _fit(self._time_pitch(stems[name] * gain, sample_rate), processed.shape[1])

        # Stage 1.5: Differential Transient Shaping (drums only when stems are available)
        if transient_on:
            update("Applying Transient Attack Shaping...", 0.35)
            if "drums" in steer:
                shaped = apply_transient_shaping(steer["drums"], sample_rate, config=self.transient_config)
                processed = processed + (shaped - steer["drums"])
                steer["drums"] = shaped
            else:
                processed = apply_transient_shaping(processed, sample_rate, config=self.transient_config)

        # Stage 1.75: De-esser (vocal sibilance only when stems are available)
        if deess_on:
            update("Applying De-esser...", 0.42)
            processed = apply_deesser(processed, sample_rate, self.deesser_config, source=steer.get("vocals"))

        # Stage 2: Psychoacoustic Sweetener (Fletcher-Munson EQ, Mono Sub, Air, Warmth)
        update("Applying Psychoacoustic Sweetening (Auto Fine-Tune)...", 0.50)
        processed = apply_sweetener(
            processed,
            sample_rate,
            config=self.sweetener_config
        )

        # Stage 3: Abbey Road Reverb & Dynamic Ducking (if configured)
        if reverb_on:
            update("Applying Abbey Road Dynamic Reverb...", 0.75)
            send = None
            if all(name in steer for name in STEM_REVERB_SEND):
                send = sum(level * steer[name] for name, level in STEM_REVERB_SEND.items())
            processed = apply_abbey_road_reverb(
                processed,
                sample_rate,
                config=self.reverb_config,
                send=send
            )

        # Stage 4: Studio Mastering & True-Peak Brickwall Limiting
        update("Finalizing Studio Mastering & Limiter...", 0.90)
        processed = master_audio(
            processed,
            sample_rate,
            config=self.mastering_config
        )

        update("Remix Complete!", 1.0)
        return processed
