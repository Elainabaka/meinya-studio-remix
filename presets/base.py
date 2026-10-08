"""
Abstract Base Class for Remix Presets (Plug & Play Architecture)
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import numpy as np
from core.pipeline import RemixPipeline


class RemixPreset(ABC):
    """
    Every remix preset must inherit from this class.
    Adding a new preset only requires implementing name, slug, description, and build_pipeline().
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable display name."""
        pass

    @property
    @abstractmethod
    def slug(self) -> str:
        """Unique CLI identifier (e.g. 'slowed_reverb')."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Short description of the preset sound character."""
        pass

    @abstractmethod
    def build_pipeline(self, **kwargs) -> RemixPipeline:
        """Constructs and returns the configured RemixPipeline."""
        pass

    def build(self, knobs: Optional[dict] = None, **kwargs) -> RemixPipeline:
        """Preset pipeline with optional shared knob overrides (see RemixPipeline.set_knobs)."""
        pipeline = self.build_pipeline(**kwargs)
        if knobs:
            pipeline.set_knobs(**knobs)
        return pipeline

    def apply(
        self,
        audio: np.ndarray,
        sample_rate: int,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        knobs: Optional[dict] = None,
        stems: Optional[dict] = None,
        **kwargs
    ) -> np.ndarray:
        """Executes the preset on the input audio (stems: optional core.stems output of the same mix)."""
        pipeline = self.build(knobs, **kwargs)
        return pipeline.process(audio, sample_rate, progress_callback=progress_callback, stems=stems)
