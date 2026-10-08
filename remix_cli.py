#!/usr/bin/env python3
"""
Studio Remix CLI: Ultra-practical, high-fidelity audio remixing tool.
Supports single-file & batch folder processing, all audio formats (MP3/WAV/FLAC/M4A/OPUS),
and real-time psychoacoustic mastering.
"""

import os
import sys
import argparse
import glob
import time
import io

# Ensure UTF-8 output encoding on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add script directory to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.audio_io import load_audio, save_audio
from presets.registry import list_presets, get_preset, register_preset

STEM_CACHE_DIR = os.path.join(SCRIPT_DIR, ".studio_cache", "stems")
RESTORE_CACHE_DIR = os.path.join(SCRIPT_DIR, ".studio_cache", "restore")


BANNER = r"""
=============================================================================
   ___ _   _ ____ ___ ___   ____  _____ __  __ _____  __  _   _ _   _    _    ____ 
  / ___| | | |  _ \_ _/ _ \ |  _ \| ____|  \/  |_ _\ \/ / | \ | | | | |  / \  / ___|
 | |   | | | | |_) | | | | || |_) |  _| | |\/| || | \  /  |  \| | |_| | / _ \| |    
 | |___| |_| |  _ <| | |_| ||  _ <| |___| |  | || | /  \  | |\  |  _  |/ ___ \ |___ 
  \____|\___/|_| \_\___\___/ |_| \_\_____|_|  |_|___/_/\_\ |_| \_|_| |_/_/   \_\____|
                     HIGH-FIDELITY DSP STUDIO ENGINE (v1.0.0)
=============================================================================
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Studio-Grade Audio Remix & Sweetener Tool",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "input",
        nargs="?",
        default=None,
        help="Path to an audio file (MP3, WAV, FLAC, M4A, OPUS, etc.) or directory for batch processing."
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Path to output audio file or output directory."
    )
    parser.add_argument(
        "-p", "--preset",
        default="slowed_reverb",
        help="Remix preset to apply (use --list to see all options). Default: slowed_reverb"
    )
    parser.add_argument(
        "-s", "--speed",
        type=float,
        default=None,
        help="Speed multiplier override (e.g. 0.85, 1.25)."
    )
    parser.add_argument(
        "--pitch",
        type=float,
        default=None,
        help="Pitch shift in semitones (e.g. -2.0, +3.0). Setting it switches to key-lock\n"
             "(Rubber Band, independent of speed). Default: vinyl, pitch follows speed."
    )
    parser.add_argument(
        "-r", "--reverb",
        type=float,
        default=None,
        help="Reverb wet level override (0.0 to 1.0, e.g. 0.35)."
    )
    parser.add_argument(
        "-b", "--bass",
        type=float,
        default=None,
        help="Bass boost dB override (e.g. 3.0, 6.0)."
    )
    parser.add_argument(
        "-f", "--format",
        default=None,
        choices=["mp3", "wav", "flac"],
        help="Output audio format. Default preserves input format or uses mp3."
    )
    parser.add_argument(
        "--bitrate",
        default="320k",
        help="MP3 export bitrate (e.g. '320k', '256k'). Default: 320k"
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="List all registered presets and exit."
    )
    parser.add_argument(
        "--stems",
        action="store_true",
        help="Stem-aware mode (AI separation, needs torch + demucs): reverb only on vocals/melody,\n"
             "drums & bass stay dry, punch on drums only, de-ess on vocals only. Cached per song."
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Bản công khai không có tính năng này. AI high-band restoration (Apollo): rebuilds what an MP3-like codec cut off above ~16 kHz.\n"
             "Skipped by itself when the source is full band. Cached per song."
    )

    return parser.parse_args()


def display_preset_list():
    presets = list_presets()
    print("\n[AVAILABLE REMIX PRESETS]")
    print("-" * 75)
    for p in presets:
        print(f"  * {p.slug:<15} : {p.name}")
        print(f"    {'':<17}   {p.description}")
    print("-" * 75)


def process_single_file(
    input_path: str,
    output_path: str,
    preset_slug: str,
    speed: float = None,
    pitch: float = None,
    reverb: float = None,
    bass: float = None,
    bitrate: str = "320k",
    stems: bool = False,
    restore: bool = False
):
    if os.path.normcase(os.path.realpath(input_path)) == os.path.normcase(os.path.realpath(output_path)):
        raise ValueError("Output must be different from the source audio")
    preset = get_preset(preset_slug)
    if preset is None:
        print(f"[!] Error: Preset '{preset_slug}' not recognized. Use --list to see options.")
        sys.exit(1)

    print(f"\n[+] Input file  : {input_path}")
    print(f"[+] Preset      : {preset.name} ({preset.slug})")

    t_start = time.perf_counter()

    # Load audio
    print(f"[*] Loading audio stream...")
    audio, sr = load_audio(input_path)
    in_duration = audio.shape[1] / sr
    print(f"[*] Loaded {in_duration:.1f}s | Sample Rate: {sr} Hz | Channels: {audio.shape[0]}")

    # Build kwargs overrides
    kwargs = {}
    if speed is not None:
        kwargs["speed"] = speed
    if pitch is not None:
        kwargs["pitch_semitones"] = pitch
    if reverb is not None:
        kwargs["reverb_amount"] = reverb
    if bass is not None:
        kwargs["bass_boost_db"] = bass

    # Execution callback
    def on_progress(stage: str, frac: float):
        pct = int(frac * 100)
        bar = ("#" * (pct // 4)).ljust(25, "-")
        print(f"\r    [{bar}] {pct:>3}% -> {stage}", end="", flush=True)

    print(f"[*] Processing DSP chain...")
    stem_audio = None
    if stems:
        from core.stems import stems_available, cached_stems
        if not stems_available():
            print("[!] --stems needs: pip install demucs (and torch)")
            sys.exit(1)
        stem_audio = cached_stems(input_path, audio, sr, STEM_CACHE_DIR, progress_callback=on_progress)
        print()
    if restore:
        from core.restore import restore_available, cached_restore, source_cutoff
        cutoff = source_cutoff(audio, sr)
        if cutoff is None:
            print("[*] --restore: source is full band, nothing to restore")
        elif not restore_available():
            print("[!] --restore: bản công khai không có tính năng này, bỏ qua bước bù dải cao")
        else:
            print(f"[*] Source cut at {cutoff / 1000:.1f} kHz: restoring the band above")
            audio = audio + cached_restore(input_path, audio, sr, RESTORE_CACHE_DIR, progress_callback=on_progress)
            print()

    remixed = preset.apply(audio, sr, progress_callback=on_progress, stems=stem_audio, **kwargs)
    print()

    # Save output
    print(f"[*] Exporting to: {output_path}")
    save_audio(output_path, remixed, sr, bitrate=bitrate)

    t_elapsed = time.perf_counter() - t_start
    out_duration = remixed.shape[1] / sr
    rtf = t_elapsed / out_duration

    print(f"[OK] Success! Exported in {t_elapsed:.2f}s ({rtf:.2f}x Real-Time)")
    print(f"[OK] Output duration: {out_duration:.1f}s | File: {output_path}\n")


def interactive_mode():
    print(BANNER)
    print("Welcome to Studio Remix CLI (Interactive Mode)\n")
    display_preset_list()

    input_path = input("\nEnter input audio file or folder path: ").strip().strip('"').strip("'")
    if not input_path or not os.path.exists(input_path):
        print(f"[!] Path does not exist: {input_path}")
        return

    preset_choice = input("Enter preset slug (default: slowed_reverb): ").strip()
    if not preset_choice:
        preset_choice = "slowed_reverb"

    if os.path.isfile(input_path):
        base, ext = os.path.splitext(input_path)
        default_out = f"{base}_{preset_choice}{ext}"
        out_path = input(f"Output path (default: {os.path.basename(default_out)}): ").strip().strip('"').strip("'")
        if not out_path:
            out_path = default_out
        process_single_file(input_path, out_path, preset_choice)
    else:
        # Directory batch
        batch_process(input_path, None, preset_choice)


def batch_process(
    in_dir: str,
    out_dir: str = None,
    preset_slug: str = "slowed_reverb",
    out_format: str = "mp3",
    **overrides
):
    extensions = ("*.mp3", "*.wav", "*.flac", "*.m4a", "*.ogg", "*.aac", "*.opus")
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(in_dir, ext)))
        files.extend(glob.glob(os.path.join(in_dir, ext.upper())))
    # Windows globs case-insensitively, so *.mp3 and *.MP3 return the same file twice.
    # Dedupe on the normalized path but keep the original spelling for output names.
    files = sorted({os.path.normcase(os.path.abspath(f)): f for f in files}.values())

    if not files:
        print(f"[!] No audio files found in directory: {in_dir}")
        return

    if out_dir is None:
        out_dir = os.path.join(in_dir, f"remix_{preset_slug}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n[BATCH PROCESSING] Found {len(files)} tracks in {in_dir}")
    print(f"Output directory: {out_dir}")

    for idx, fpath in enumerate(files, 1):
        fname = os.path.basename(fpath)
        base = os.path.splitext(fname)[0]
        ext = f".{out_format}" if out_format else os.path.splitext(fname)[1]
        out_path = os.path.join(out_dir, f"{base}_{preset_slug}{ext}")
        print(f"\n--- Track [{idx}/{len(files)}]: {fname} ---")
        try:
            process_single_file(
                fpath,
                out_path,
                preset_slug,
                **overrides
            )
        except Exception as e:
            print(f"[!] Error processing {fname}: {e}")


def main():
    args = parse_args()

    if args.list:
        print(BANNER)
        display_preset_list()
        return

    if args.input is None:
        interactive_mode()
        return

    print(BANNER)

    # Batch directory or single file
    if os.path.isdir(args.input):
        batch_process(
            args.input,
            out_dir=args.output,
            preset_slug=args.preset,
            out_format=args.format or "mp3",
            speed=args.speed,
            pitch=args.pitch,
            reverb=args.reverb,
            bass=args.bass,
            bitrate=args.bitrate,
            stems=args.stems,
            restore=args.restore
        )
    elif os.path.isfile(args.input):
        out_path = args.output
        if out_path is None:
            base, ext = os.path.splitext(args.input)
            ext_choice = f".{args.format}" if args.format else ext
            out_path = f"{base}_{args.preset}{ext_choice}"
        
        process_single_file(
            args.input,
            out_path,
            args.preset,
            speed=args.speed,
            pitch=args.pitch,
            reverb=args.reverb,
            bass=args.bass,
            bitrate=args.bitrate,
            stems=args.stems,
            restore=args.restore
        )
    else:
        print(f"[!] Error: Input path does not exist: {args.input}")
        sys.exit(1)


if __name__ == "__main__":
    main()
