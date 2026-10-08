"""
Synthetic Signal Generator for Audio DSP Testing & Benchmarking
Generates mathematically clean test signals: Chirps, Drum Transients, Chords, and Full Synthetic Mix.
"""

import numpy as np
from scipy import signal


def create_chirp_signal(sample_rate: int = 44100, duration_sec: float = 2.0) -> np.ndarray:
    """
    Logarithmic frequency sweep from 20 Hz to 20,000 Hz.
    Useful for testing full-spectrum EQ response and filter stability.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    chirp = signal.chirp(t, f0=20, t1=duration_sec, f1=20000, method='logarithmic')
    chirp = (chirp * 0.5).astype(np.float32)
    return np.stack([chirp, chirp], axis=0)


def create_drum_transient_loop(sample_rate: int = 44100, duration_sec: float = 3.0) -> np.ndarray:
    """
    Generates sharp percussive transients (synthetic kick, snare, hi-hat)
    to test time-stretching transient preservation and sidechain ducking.
    """
    num_samples = int(sample_rate * duration_sec)
    audio = np.zeros(num_samples, dtype=np.float32)
    
    # 120 BPM: Beat interval = 0.5 seconds
    beat_samples = int(sample_rate * 0.5)
    
    for i in range(0, num_samples, beat_samples):
        # Synthetic Kick (sine pitch drop 150Hz -> 45Hz)
        kick_len = min(int(sample_rate * 0.15), num_samples - i)
        t_k = np.linspace(0, 0.15, kick_len, endpoint=False)
        freq_k = 150 * np.exp(-t_k * 25) + 45
        kick = np.sin(2 * np.pi * freq_k * t_k) * np.exp(-t_k * 20)
        audio[i:i + kick_len] += kick * 0.8
        
        # Synthetic Snare at offset
        snare_pos = i + beat_samples // 2
        if snare_pos < num_samples:
            snare_len = min(int(sample_rate * 0.2), num_samples - snare_pos)
            t_s = np.linspace(0, 0.2, snare_len, endpoint=False)
            noise = np.random.uniform(-1, 1, snare_len) * np.exp(-t_s * 30)
            tone = np.sin(2 * np.pi * 200 * t_s) * np.exp(-t_s * 25)
            audio[snare_pos:snare_pos + snare_len] += (noise * 0.5 + tone * 0.5) * 0.7

    audio = np.clip(audio, -0.9, 0.9)
    return np.stack([audio, audio], axis=0)


def create_synthetic_music_mix(sample_rate: int = 44100, duration_sec: float = 4.0) -> np.ndarray:
    """
    Combines melodic chord progression, bassline, and drums for full pipeline testing.
    """
    num_samples = int(sample_rate * duration_sec)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)

    # Rich chord progression (A minor -> F major -> C major -> G major)
    melody = (
        0.20 * np.sin(2 * np.pi * 440.0 * t) +   # A4
        0.15 * np.sin(2 * np.pi * 523.25 * t) +  # C5
        0.15 * np.sin(2 * np.pi * 659.25 * t)    # E5
    )

    # Sub-bassline at 55 Hz (A1)
    bass = 0.35 * np.sin(2 * np.pi * 55.0 * t)

    # Drums
    drums = create_drum_transient_loop(sample_rate, duration_sec)[0]

    # Stereo placement: Bass centered, melody spread
    left = (melody * 0.8 + bass * 0.5 + drums * 0.5).astype(np.float32)
    right = (melody * 0.9 + bass * 0.5 + drums * 0.5).astype(np.float32)

    mix = np.stack([left, right], axis=0)
    # Normalize to -3 dBFS
    max_val = np.max(np.abs(mix))
    if max_val > 0:
        mix = mix / max_val * 0.7
    return mix
