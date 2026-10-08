"""
Integrated loudness (ITU-R BS.1770-4) with the exact numbers of pyloudnorm.Meter.integrated_loudness, faster.

pyloudnorm K-weights one channel at a time (float64 lfilter, stored back into a copy of the input dtype
between the two stages), then sums each 400 ms block with its own np.sum inside a Python loop.
This module does the same arithmetic in the same order, so the result is bit-identical (test_17):
the channels are filtered in parallel threads and the block sums run as a few batched reductions.
"""
from concurrent.futures import ThreadPoolExecutor
import warnings

import numpy as np
import pyloudnorm as pyln
from scipy import signal

_G = (1.0, 1.0, 1.0, 1.41, 1.41)  # BS.1770 channel gains (L, R, C, Ls, Rs)
_T_G = 0.400                       # gating block (s)
_STEP = 1.0 - 0.75                 # 75 % overlap
_GAMMA_A = -70.0                   # absolute gate (LUFS)
_BATCH = 256                       # blocks per batched reduction (bounds the temporary memory)
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lufs")


def _filters(rate: int):
    return list(pyln.Meter(rate)._filters.values())


def _k_weight(x: np.ndarray, filters) -> np.ndarray:
    for f in filters:
        x = (f.passband_gain * signal.lfilter(f.b, f.a, x)).astype(x.dtype)
    return x


def _block_sums(sq: np.ndarray, lo: np.ndarray, length: int) -> np.ndarray:
    """np.sum(sq[l:l+length]) for every l (sq = the squared signal), as row reductions of strided views:
    no copy, and each row is one contiguous pairwise sum like the 1-D np.sum it replaces."""
    out = np.zeros(len(lo), dtype=sq.dtype)
    if length <= 0:
        return out
    # Block starts form an arithmetic progression (hop) at the usual rates; float rounding in pyloudnorm's
    # expressions can move a few by one sample at odd rates: those rows are summed one by one.
    anchor = int(lo[0])
    hop = max(1, int(round((lo[-1] - anchor) / (len(lo) - 1)))) if len(lo) > 1 else 1
    on_grid = lo == anchor + np.arange(len(lo)) * hop
    if on_grid.any():
        n_rows = int(np.nonzero(on_grid)[0][-1]) + 1
        view = np.lib.stride_tricks.as_strided(sq[anchor:], shape=(n_rows, length),
                                               strides=(hop * sq.strides[0], sq.strides[0]), writeable=False)
        out[:n_rows] = view.sum(axis=1)
    for r in np.nonzero(~on_grid)[0]:
        out[r] = np.sum(sq[lo[r]:lo[r] + length])
    return out


def integrated_loudness(audio: np.ndarray, rate: int) -> float:
    """audio: (channels, samples) float array. Same value as pyln.Meter(rate).integrated_loudness(audio.T);
    raises ValueError for clips shorter than one block, like pyloudnorm."""
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if not np.issubdtype(audio.dtype, np.floating):
        raise ValueError("Data must be floating point.")
    if audio.shape[0] > 5:
        raise ValueError("Audio must have five channels or less.")
    num_ch, num_samples = audio.shape
    if num_samples < _T_G * rate:
        raise ValueError("Audio must have length greater than the block size.")

    filters = _filters(rate)
    weighted = list(_pool.map(lambda ch: _k_weight(audio[ch], filters), range(num_ch)))

    T = num_samples / rate
    num_blocks = int(np.round(((T - _T_G) / (_T_G * _STEP)))) + 1
    j = np.arange(num_blocks)
    lo = (_T_G * (j * _STEP) * rate).astype(np.int64)      # int() of the same float expressions as pyloudnorm
    hi = (_T_G * (j * _STEP + 1) * rate).astype(np.int64)
    hi = np.minimum(hi, num_samples)                        # slices past the end are clipped, as in numpy slicing
    z = np.zeros((num_ch, num_blocks))
    # pyloudnorm multiplies a Python float by the block sum, which NumPy 2 does in the sum's dtype
    scale = weighted[0].dtype.type(1.0 / (_T_G * rate))
    squared = [np.square(w) for w in weighted]
    for length in np.unique(hi - lo):
        sel = np.nonzero(hi - lo == length)[0]
        for i in range(num_ch):
            z[i, sel] = _block_sums(squared[i], lo[sel], int(length)) * scale

    # Gating: kept verbatim from pyloudnorm (a few thousand scalar ops, and the exact same rounding)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        l = [-0.691 + 10.0 * np.log10(np.sum([_G[i] * z[i, jj] for i in range(num_ch)])) for jj in j]
        J_g = [jj for jj, l_j in enumerate(l) if l_j >= _GAMMA_A]
        z_avg_gated = [np.mean([z[i, jj] for jj in J_g]) for i in range(num_ch)]
        Gamma_r = -0.691 + 10.0 * np.log10(np.sum([_G[i] * z_avg_gated[i] for i in range(num_ch)])) - 10.0
        J_g = [jj for jj, l_j in enumerate(l) if (l_j > Gamma_r and l_j > _GAMMA_A)]
        z_avg_gated = np.nan_to_num(np.array([np.mean([z[i, jj] for jj in J_g]) for i in range(num_ch)]))
    with np.errstate(divide="ignore"):
        return -0.691 + 10.0 * np.log10(np.sum([_G[i] * z_avg_gated[i] for i in range(num_ch)]))
