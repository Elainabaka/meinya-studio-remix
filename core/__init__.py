"""
Core DSP Engine for High-Fidelity Audio Remixing & Sweetening
"""

from .audio_io import load_audio, save_audio
from .time_pitch import time_pitch_process
from .transient import apply_transient_shaping, TransientConfig
from .deesser import apply_deesser, DeEsserConfig
from .sweetener import apply_sweetener, SweetenerConfig
from .reverb_engine import apply_abbey_road_reverb, ReverbConfig
from .mastering import master_audio, MasteringConfig
from .pipeline import RemixPipeline

__all__ = [
    "load_audio",
    "save_audio",
    "time_pitch_process",
    "apply_transient_shaping",
    "TransientConfig",
    "apply_deesser",
    "DeEsserConfig",
    "apply_sweetener",
    "SweetenerConfig",
    "apply_abbey_road_reverb",
    "ReverbConfig",
    "master_audio",
    "MasteringConfig",
    "RemixPipeline",
]
