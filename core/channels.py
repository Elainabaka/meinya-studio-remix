"""
Run a stage on each channel in its own thread, for stages that treat L and R independently.

Checked bit for bit on a real song driven from -6 dB to +10 dB (PERF_NOTES.md): pedalboard filters, EQ and
Compressor give the same samples on L and R alone as on the stereo pair. Reverb (mixes channels) and the
true-peak BrickwallLimiter (stereo-linked once it limits) do not, and must never go through here (test_18).
pedalboard and scipy's filters release the GIL, so 2 threads use 2 cores.
"""
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np

_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="dsp-channel")
MIN_SAMPLES = 48000  # below ~1 s the thread hand-off costs more than it saves


def per_channel(fn: Callable[[np.ndarray], np.ndarray], audio: np.ndarray) -> np.ndarray:
    """fn maps a (1, n) array to a (1, m) array and must not share state between calls
    (build a fresh pedalboard per call). Returns the channels stacked back in order."""
    if audio.ndim != 2 or audio.shape[0] < 2 or audio.shape[1] < MIN_SAMPLES:
        return fn(audio)
    parts = list(_pool.map(lambda c: fn(np.ascontiguousarray(audio[c:c + 1])), range(audio.shape[0])))
    return np.concatenate(parts, axis=0)
