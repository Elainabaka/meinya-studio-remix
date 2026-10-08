"""
Presets Package for High-Fidelity Audio Remixing
"""

from .base import RemixPreset
from .registry import register_preset, get_preset, list_presets
from .slowed_reverb import SlowedReverbPreset
from .nightcore import NightcorePreset
from .bass_boost import BassBoostPreset
from .lofi_chill import LoFiChillPreset
from .spatial_8d import Spatial8DPreset
from .sped_up import SpedUpPreset
from .vaporwave import VaporwavePreset

__all__ = [
    "RemixPreset",
    "register_preset",
    "get_preset",
    "list_presets",
    "SlowedReverbPreset",
    "NightcorePreset",
    "BassBoostPreset",
    "LoFiChillPreset",
    "Spatial8DPreset",
    "SpedUpPreset",
    "VaporwavePreset",
]
