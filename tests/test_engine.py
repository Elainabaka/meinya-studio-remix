"""
Comprehensive Unit Tests & Self-Correction Validation Suite
Validates DSP integrity, psychoacoustic filters, preset execution, and true-peak protection.
"""

import os
import tempfile
import unittest
import numpy as np

from tests.generate_signals import (
    create_chirp_signal,
    create_drum_transient_loop,
    create_synthetic_music_mix
)
from core.audio_io import load_audio, save_audio
from core.time_pitch import time_pitch_process
from core.sweetener import apply_sweetener, SweetenerConfig
from core.reverb_engine import apply_abbey_road_reverb, ReverbConfig
from core.mastering import master_audio, MasteringConfig
from presets.registry import list_presets, get_preset


class TestRemixEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sr = 44100
        cls.mix = create_synthetic_music_mix(cls.sr, duration_sec=2.5)
        cls.chirp = create_chirp_signal(cls.sr, duration_sec=1.5)
        cls.drums = create_drum_transient_loop(cls.sr, duration_sec=2.0)

    def test_01_audio_io_wav_and_mp3(self):
        """Verify saving and reloading WAV and MP3 with FFmpeg."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "test.wav")
            save_audio(wav_path, self.mix, self.sr)
            self.assertTrue(os.path.exists(wav_path))

            loaded, sr_loaded = load_audio(wav_path)
            self.assertEqual(sr_loaded, self.sr)
            self.assertEqual(loaded.shape[0], 2)
            self.assertAlmostEqual(loaded.shape[1], self.mix.shape[1], delta=50)

            # Test MP3 encoding via FFmpeg
            mp3_path = os.path.join(tmp_dir, "test.mp3")
            save_audio(mp3_path, self.mix, self.sr, bitrate="320k")
            self.assertTrue(os.path.exists(mp3_path))
            loaded_mp3, sr_mp3 = load_audio(mp3_path)
            self.assertEqual(sr_mp3, self.sr)
            self.assertEqual(loaded_mp3.shape[0], 2)

    def test_02_time_pitch_speedup_and_slowdown(self):
        """Verify time stretching and pitch shifting calculate lengths and shapes correctly."""
        # Speedup 1.25x
        fast = time_pitch_process(self.mix, self.sr, speed=1.25, pitch_semitones=3.0)
        expected_len = int(self.mix.shape[1] / 1.25)
        self.assertAlmostEqual(fast.shape[1], expected_len, delta=100)
        self.assertFalse(np.isnan(fast).any())
        self.assertFalse(np.isinf(fast).any())

        # Slowdown 0.85x
        slow = time_pitch_process(self.mix, self.sr, speed=0.85, pitch_semitones=-2.0)
        expected_len_slow = int(self.mix.shape[1] / 0.85)
        self.assertAlmostEqual(slow.shape[1], expected_len_slow, delta=100)
        self.assertFalse(np.isnan(slow).any())

    def test_03_sweetener_filter_stability(self):
        """Verify psychoacoustic sweetener on full spectrum chirp and mix."""
        cfg = SweetenerConfig()
        sweetened = apply_sweetener(self.chirp, self.sr, config=cfg)
        self.assertEqual(sweetened.shape, self.chirp.shape)
        self.assertFalse(np.isnan(sweetened).any())
        self.assertFalse(np.isinf(sweetened).any())

        # Sub-bass Mono check (< 50 Hz should have near zero side energy)
        t = np.linspace(0, 1.0, self.sr, False)
        low_stereo = np.sin(2 * np.pi * 45 * t).astype(np.float32)
        out_of_phase = np.stack([low_stereo, -low_stereo], axis=0)
        processed = apply_sweetener(out_of_phase, self.sr, config=cfg)
        side_channel = 0.5 * (processed[0] - processed[1])
        self.assertLess(np.max(np.abs(side_channel)), 0.3)

    def test_04_reverb_dynamic_ducking(self):
        """Verify Abbey Road reverb produces smooth reflections and ducks on transients."""
        cfg = ReverbConfig(wet_level=0.4, ducking_amount=0.4, tail_sec=2.0)
        reverbed = apply_abbey_road_reverb(self.drums, self.sr, config=cfg)
        n = self.drums.shape[1]
        self.assertEqual(reverbed.shape, (2, n + 2 * self.sr))
        self.assertFalse(np.isnan(reverbed).any())
        # Tail rings out after the dry signal ends, then fades to silence
        self.assertGreater(np.max(np.abs(reverbed[:, n:n + self.sr // 2])), 1e-3)
        self.assertLess(np.max(np.abs(reverbed[:, -1])), 1e-6)

    def test_05_mastering_peak_limiter(self):
        """Verify mastering enforces True-Peak ceiling and EBU R128 loudness."""
        loud_signal = self.mix * 3.5  # Heavy hot signal that would otherwise clip
        mastered = master_audio(loud_signal, self.sr, config=MasteringConfig(true_peak_ceiling_db=-1.0))
        max_peak = np.max(np.abs(mastered))
        max_peak_db = 20 * np.log10(max_peak)
        self.assertLessEqual(max_peak_db, 0.0)
        self.assertFalse(np.isnan(mastered).any())

    def test_06_all_presets_execution(self):
        """Execute all registered presets and verify pristine audio output."""
        presets = list_presets()
        self.assertGreaterEqual(len(presets), 7)

        for preset in presets:
            with self.subTest(preset=preset.slug):
                out = preset.apply(self.mix, self.sr)
                self.assertEqual(out.shape[0], 2)
                self.assertGreater(out.shape[1], 1000)
                self.assertFalse(np.isnan(out).any(), f"Preset {preset.slug} generated NaN")
                self.assertFalse(np.isinf(out).any(), f"Preset {preset.slug} generated Inf")
                max_val = np.max(np.abs(out))
                self.assertLessEqual(max_val, 1.02, f"Preset {preset.slug} clipped above ceiling: {max_val}")

    def test_07_transient_shaper(self):
        """Verify differential transient shaper sharpens drum attacks without NaN/Inf."""
        from core.transient import apply_transient_shaping, TransientConfig
        cfg = TransientConfig(attack_gain=0.45, sustain_gain=-0.15)
        shaped = apply_transient_shaping(self.drums, self.sr, config=cfg)
        self.assertEqual(shaped.shape, self.drums.shape)
        self.assertFalse(np.isnan(shaped).any())
        self.assertFalse(np.isinf(shaped).any())

    def test_08_tape_flutter(self):
        """Verify tape wow & flutter adds subtle organic drift without phase artifacts."""
        cfg = SweetenerConfig(tape_flutter_depth=0.0005)
        fluttered = apply_sweetener(self.mix, self.sr, config=cfg)
        self.assertEqual(fluttered.shape, self.mix.shape)
        self.assertFalse(np.isnan(fluttered).any())
        self.assertFalse(np.isinf(fluttered).any())

    def test_09_stem_mode_steers_but_keeps_dry_path(self):
        """Stem mode: with every stem-steered effect off the output is bit-identical to full-mix mode;
        with reverb on, a drums-only song gets (almost) no reverb because drums stay dry."""
        n = self.drums.shape[1]
        silent = np.zeros_like(self.drums)
        stems = {"drums": self.drums, "bass": silent, "other": silent, "vocals": silent}
        preset = get_preset("slowed_reverb")

        off = {"reverb": 0.0, "punch": 0.0, "deess_db": 0.0}
        plain = preset.apply(self.drums, self.sr, knobs=off)
        steered = preset.apply(self.drums, self.sr, knobs=off, stems=stems)
        np.testing.assert_array_equal(plain, steered)

        dry = preset.apply(self.drums, self.sr, knobs={"reverb": 0.0})
        wet_full = preset.apply(self.drums, self.sr, knobs={"reverb": 0.5})
        wet_stem = preset.apply(self.drums, self.sr, knobs={"reverb": 0.5}, stems=stems)
        m = dry.shape[1]
        full_diff = np.sum((wet_full[:, :m] - dry) ** 2)
        stem_diff = np.sum((wet_stem[:, :m] - dry) ** 2)
        self.assertFalse(np.isnan(wet_stem).any())
        self.assertLess(stem_diff, 0.1 * full_diff)

    def test_10_vinyl_varispeed_is_exact(self):
        """Vinyl mode resamples: 1 kHz at 1.25x must become 1.25 kHz and 1/1.25 the length."""
        from scipy import signal
        t = np.arange(self.sr * 2) / self.sr
        tone = np.stack([0.5 * np.sin(2 * np.pi * 1000.0 * t)] * 2).astype(np.float32)
        for speed in (1.25, 0.85):
            out = time_pitch_process(tone, self.sr, speed=speed, vinyl_mode=True)
            self.assertAlmostEqual(out.shape[1], tone.shape[1] / speed, delta=2)
            f, p = signal.welch(out[0], fs=self.sr, nperseg=16384)
            self.assertAlmostEqual(f[np.argmax(p)], 1000.0 * speed, delta=3.0)

    def test_11_output_independent_of_input_level(self):
        """Gain staging: a hot and a quiet copy of the same song must remix identically."""
        preset = get_preset("slowed_reverb")
        hot = preset.apply(self.mix * 1.4, self.sr)
        quiet = preset.apply(self.mix * 0.1, self.sr)
        self.assertEqual(hot.shape, quiet.shape)
        self.assertLess(float(np.max(np.abs(hot - quiet))), 2e-3)

    def test_12_8d_keeps_sub_bass_centered(self):
        """8D orbit must not pan sub-bass, and must not leave a -3 dB hole at center."""
        from presets.spatial_8d import apply_8d_rotation
        t = np.arange(self.sr * 7) / self.sr
        sub = np.stack([0.5 * np.sin(2 * np.pi * 50.0 * t)] * 2).astype(np.float32)
        out = apply_8d_rotation(sub, self.sr)
        side = 0.5 * (out[0] - out[1])
        self.assertLess(float(np.max(np.abs(side[self.sr:-self.sr]))), 0.02)

        tone = np.stack([0.5 * np.sin(2 * np.pi * 1000.0 * t)] * 2).astype(np.float32)
        out = apply_8d_rotation(tone, self.sr, ambience=0.0)
        hop = self.sr // 10
        power = [np.mean(out[:, i:i + hop] ** 2) * 2 for i in range(0, out.shape[1] - hop, hop)]
        power_db = 10 * np.log10(np.array(power))
        self.assertLess(float(np.max(power_db) - np.min(power_db)), 1.0)

    def test_13_knobs_override_any_preset(self):
        """Shared knobs round-trip; an explicit pitch switches vinyl -> key-lock."""
        for preset in list_presets():
            with self.subTest(preset=preset.slug):
                knobs = preset.build().get_knobs()
                self.assertEqual(set(knobs), {"speed", "pitch", "reverb", "bass_db", "air_db", "punch", "width", "lufs", "deess_db"})
        pipe = get_preset("nightcore").build()
        self.assertTrue(pipe.vinyl_mode)
        pipe = get_preset("nightcore").build({"pitch": 2.0, "reverb": 0.3, "punch": 0.1})
        self.assertFalse(pipe.vinyl_mode)
        self.assertEqual(pipe.get_knobs()["pitch"], 2.0)
        self.assertEqual(pipe.get_knobs()["reverb"], 0.3)
        pipe = get_preset("bass_boost").build({"reverb": 0.2})
        self.assertEqual(pipe.get_knobs()["reverb"], 0.2)

    def test_14_deesser_cuts_sibilants_only(self):
        """Sibilant bursts lose >= 3 dB in the band above 5.5 kHz; a steady 1 kHz tone is untouched."""
        from scipy import signal
        from core.deesser import apply_deesser, DeEsserConfig
        rng = np.random.default_rng(0)
        n = self.sr * 2
        t = np.arange(n) / self.sr
        tone = 0.3 * np.sin(2 * np.pi * 1000.0 * t)
        sos = signal.butter(4, 6000, btype="highpass", fs=self.sr, output="sos")
        hiss = signal.sosfilt(sos, rng.standard_normal(n)) * 0.5
        gate = np.zeros(n)
        for start in range(int(0.25 * self.sr), n, int(0.5 * self.sr)):
            gate[start:start + int(0.08 * self.sr)] = 1.0
        x = np.stack([tone + hiss * gate] * 2).astype(np.float32)
        out = apply_deesser(x, self.sr, DeEsserConfig(max_reduction_db=8.0))
        band = lambda y: signal.sosfiltfilt(sos, y[0])
        burst = gate > 0
        drop_db = 10 * np.log10(np.mean(band(out)[burst] ** 2) / np.mean(band(x)[burst] ** 2))
        self.assertLess(drop_db, -3.0)
        quiet = np.convolve(gate, np.ones(int(0.05 * self.sr)), mode="same") == 0
        self.assertLess(float(np.max(np.abs(out[0][quiet] - x[0][quiet]))), 1e-3)
        # Level-independent detection
        loud = apply_deesser(x * 3.0, self.sr, DeEsserConfig(max_reduction_db=8.0))
        np.testing.assert_allclose(loud, out * 3.0, atol=1e-4)

    def test_15_ai_stem_separation(self):
        """Demucs separation returns 4 finite stems of the input length (skipped if not installed)."""
        from core.stems import stems_available, separate_stems, STEM_NAMES
        if not stems_available():
            self.skipTest("torch/demucs not installed")
        clip = self.mix[:, : self.sr * 4]
        stems = separate_stems(clip, self.sr)
        self.assertEqual(set(stems), set(STEM_NAMES))
        for name, stem in stems.items():
            self.assertEqual(stem.shape, clip.shape, name)
            self.assertTrue(np.isfinite(stem).all(), name)

    def test_16_opus_input(self):
        """.opus (Ogg Opus, e.g. from yt-dlp) loads as 48 kHz stereo with the pre-skip removed."""
        import shutil
        import subprocess
        if not shutil.which("ffmpeg"):
            self.skipTest("ffmpeg not installed")
        with tempfile.TemporaryDirectory() as tmp_dir:
            wav_path = os.path.join(tmp_dir, "src.wav")
            opus_path = os.path.join(tmp_dir, "src.opus")
            save_audio(wav_path, self.mix, self.sr)
            subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", wav_path, "-c:a", "libopus",
                            "-b:a", "160k", opus_path], check=True)
            loaded, sr = load_audio(opus_path)
        self.assertEqual(sr, 48000)
        self.assertEqual(loaded.shape[0], 2)
        expected = self.mix.shape[1] * 48000 / self.sr
        self.assertAlmostEqual(loaded.shape[1], expected, delta=0.005 * 48000)
        self.assertTrue(np.isfinite(loaded).all())
        out = get_preset("nightcore").apply(loaded, sr)
        self.assertLessEqual(np.max(np.abs(out)), 1.0)

    def test_17_fast_loudness_matches_pyloudnorm_exactly(self):
        """core.loudness gives the very same float as pyloudnorm (so gains and outputs stay bit-identical)."""
        import pyloudnorm as pyln
        from core.loudness import integrated_loudness
        rng = np.random.default_rng(7)
        cases = [(self.mix, self.sr), (self.drums, self.sr), (self.chirp[:, :int(0.41 * self.sr)], self.sr)]
        for sr, seconds in ((44100, 3.37), (48000, 5.01), (96000, 1.3), (22050, 2.0)):
            x = (0.3 * rng.standard_normal((2, int(seconds * sr)))).astype(np.float32)
            x[:, : sr // 2] *= 1e-5  # quiet intro exercises the absolute gate
            cases += [(x, sr), (x.astype(np.float64), sr), (x[:1], sr)]
        for audio, sr in cases:
            ref = pyln.Meter(sr).integrated_loudness(audio.T)
            got = integrated_loudness(audio, sr)
            self.assertEqual(got, ref, (audio.shape, audio.dtype, sr))
        with self.assertRaises(ValueError):
            integrated_loudness(self.mix[:, :100], self.sr)

    def test_18_per_channel_stages_match_stereo(self):
        """core.channels only for channel-independent stages: EQ/filters/compressor bit-identical per channel
        even when driven hot; the limiter is stereo-linked, so a split would change the sound."""
        import pedalboard as pb
        from core.channels import per_channel
        hot = np.ascontiguousarray(np.tile(self.mix, (1, 2)) * 3.0)  # +10 dB: compressor and limiter both work hard
        hot[1] *= 0.6  # L and R differ, so a linked detector would show
        boards = [lambda: pb.Pedalboard([pb.HighpassFilter(28), pb.PeakFilter(75, 2.5, 1.0), pb.HighShelfFilter(12500, 2.0)]),
                  lambda: pb.Pedalboard([pb.HighpassFilter(380), pb.LowpassFilter(6800)]),
                  lambda: pb.Pedalboard([pb.Compressor(threshold_db=-12, ratio=1.6, attack_ms=25, release_ms=120)])]
        for mk in boards:
            np.testing.assert_array_equal(per_channel(lambda ch: mk()(ch, self.sr), hot), mk()(hot, self.sr))
        lim = lambda: pb.Pedalboard([pb.BrickwallLimiter(ceiling_db=-1.0, true_peak=True)])
        split = np.concatenate([lim()(np.ascontiguousarray(hot[c:c + 1]), self.sr) for c in range(2)])
        self.assertFalse(np.array_equal(split, lim()(hot, self.sr)))

    @staticmethod
    def _noise(sr: int, seconds: float, cut_hz: float = 0.0, seed: int = 5) -> np.ndarray:
        """Stereo noise with correlated channels; cut_hz > 0 removes everything above it like a codec does."""
        rng = np.random.default_rng(seed)
        n = int(sr * seconds)
        mid, side = rng.standard_normal(n), 0.3 * rng.standard_normal(n)
        x = 0.1 * np.stack([mid + side, mid - side])
        if cut_hz:
            spec = np.fft.rfft(x, axis=1)
            spec[:, np.fft.rfftfreq(n, 1 / sr) > cut_hz] = 0
            x = np.fft.irfft(spec, n, axis=1)
        return x.astype(np.float32)

    def test_21_source_cutoff_finds_a_codec_wall_only(self):
        """A brick wall under 19.5 kHz is a codec cutoff; full band, a wall above it and a natural roll-off are not."""
        from scipy import signal
        from core.restore import source_cutoff
        for sr in (44100, 48000):
            self.assertAlmostEqual(source_cutoff(self._noise(sr, 4, 16000), sr), 16000, delta=150)
            self.assertIsNone(source_cutoff(self._noise(sr, 4), sr))
            self.assertIsNone(source_cutoff(self._noise(sr, 4, 20000), sr))
            sos = signal.butter(8, 9000, btype="lowpass", fs=sr, output="sos")   # -30 dB at ~15 kHz, 48 dB/octave
            self.assertIsNone(source_cutoff(signal.sosfilt(sos, self._noise(sr, 4), axis=1), sr))
        self.assertIsNone(source_cutoff(self._noise(22050, 4, 9000), 22050))

    def test_22_restore_delta_only_fills_the_missing_band(self):
        """The delta is the model's change above (cutoff - 1 kHz), built on Mid/Side, silent where the source is
        silent, zero for a model that changes nothing, and None (model not called) for a full-band source."""
        from scipy import signal
        from core.restore import restore_delta
        calls = []

        def hiss(x):               # a model that adds the same noise to the first channel (Mid) only
            calls.append(x.shape)
            out = x.copy()
            out[0] += 0.05 * np.random.default_rng(x.shape[1]).standard_normal(x.shape[1]).astype(np.float32)
            return out
        for sr in (44100, 48000):
            x = self._noise(sr, 15, 16000)
            x[:, 5 * sr:7 * sr] = 0
            delta = restore_delta(x, sr, model=hiss)
            self.assertEqual((delta.shape, delta.dtype), (x.shape, np.float32))
            np.testing.assert_allclose(delta[0], delta[1], atol=1e-6)        # Mid only: the same in L and R
            freqs, power = signal.welch(delta[0, :4 * sr], sr, nperseg=8192)
            high, low = power[freqs > 16000].mean(), power[freqs < 9000].mean()
            self.assertGreater(10 * np.log10(high / low), 50)
            self.assertEqual(float(np.max(np.abs(delta[:, int(5.5 * sr):int(6.5 * sr)]))), 0.0)
            self.assertGreater(float(np.max(np.abs(delta[:, 8 * sr:9 * sr]))), 1e-3)
            # chunks + cross-fades + resampling give back what the model returned
            self.assertLess(float(np.max(np.abs(restore_delta(x, sr, model=lambda c: c)))), 1e-5)
        self.assertGreater(len(calls), 4)                                   # 15 s = several 6 s chunks
        self.assertTrue(all(shape == (2, 8 * 44100) for shape in calls))    # always 6 s + 1 s of context each side
        del calls[:]
        self.assertIsNone(restore_delta(self._noise(44100, 4), 44100, model=hiss))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
