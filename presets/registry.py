"""
Preset Registry: Central registry for plug-and-play remix presets.
"""

from typing import Dict, List, Optional
from .base import RemixPreset
from .slowed_reverb import SlowedReverbPreset
from .nightcore import NightcorePreset
from .bass_boost import BassBoostPreset
from .lofi_chill import LoFiChillPreset
from .spatial_8d import Spatial8DPreset
from .sped_up import SpedUpPreset
from .vaporwave import VaporwavePreset

_REGISTRY: Dict[str, RemixPreset] = {}


def register_preset(preset: RemixPreset) -> None:
    """Register a preset instance into the global registry."""
    _REGISTRY[preset.slug] = preset


def get_preset(slug: str) -> Optional[RemixPreset]:
    """Retrieve a preset by its unique slug."""
    return _REGISTRY.get(slug)


def list_presets() -> List[RemixPreset]:
    """Returns a list of all currently registered presets."""
    return list(_REGISTRY.values())


# Auto-register core presets
register_preset(SlowedReverbPreset())
register_preset(NightcorePreset())
register_preset(BassBoostPreset())
register_preset(LoFiChillPreset())
register_preset(Spatial8DPreset())
register_preset(SpedUpPreset())
register_preset(VaporwavePreset())
