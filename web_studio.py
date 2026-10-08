#!/usr/bin/env python3
"""
Remix Studio - local web UI (stdlib http.server, no extra dependencies).

- Drag & drop a song or pick one from `nhạc test/`
- Pick a preset, fine-tune the shared knobs (RemixPipeline.set_knobs)
- 30 s preview of the loudest section (auto-rerender while tuning) or full export to `output/`
- Optional stem-aware mode (AI separation via core.stems, cached per song)
- Loudness-matched, tempo-aligned A/B between original and remix, plus pin + blind A/B of two remixes

Run:  python web_studio.py [port]   or   python remix_cli.py --web
"""

import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import tempfile
import urllib.parse
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from scipy import signal

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from core.audio_io import load_audio, save_audio
from core.loudness import integrated_loudness
from core import features
from core.restore import source_cutoff
from presets.registry import list_presets, get_preset

WEB_DIR = os.path.join(SCRIPT_DIR, "web")
LIBRARY_DIR = os.path.join(SCRIPT_DIR, "nhạc test")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
CACHE_DIR = os.path.join(SCRIPT_DIR, ".studio_cache")
UPLOAD_DIR = os.path.join(CACHE_DIR, "uploads")
PREVIEW_DIR = os.path.join(CACHE_DIR, "previews", uuid.uuid4().hex)
PINNED_DIR = os.path.join(CACHE_DIR, "pinned", uuid.uuid4().hex)
STEM_DIR = os.path.join(CACHE_DIR, "stems")
RESTORE_DIR = os.path.join(CACHE_DIR, "restore")

AUDIO_EXTS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus"}
MIME = {
    ".wav": "audio/wav", ".mp3": "audio/mpeg", ".flac": "audio/flac", ".m4a": "audio/mp4",
    ".ogg": "audio/ogg", ".aac": "audio/aac", ".opus": "audio/ogg",
    ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".webp": "image/webp", ".ttf": "font/ttf", ".woff2": "font/woff2", ".mp4": "video/mp4",
}
STATIC_FILES = {"index.html", "studio.js", "studio.css", "video_tab.js"}
# Tab 2 loads the same files as the renderer (web/video/template.js, fonts, chibi, logo): D2 preview = render
ROOT_SERVED = {"web": {".js", ".html"}, "brand": {".ttf", ".woff2"}, "assets": {".png", ".jpg", ".jpeg", ".webp"}}
ART_DIR = os.path.join(CACHE_DIR, "art")
PROJECT_DIR = os.path.join(CACHE_DIR, "projects")
VIDEO_OUT_DIR = os.path.join(OUTPUT_DIR, "video")
ART_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
REMIX_EXTS = {".wav", ".mp3", ".flac"}
PREVIEW_SEC = 30.0
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
PEAK_BUCKETS = 1200

# ---------------------------------------------------------------- state
_media = {}          # media_id -> absolute file path (only these are ever served)
_tracks = {}         # track_id -> absolute file path
_jobs = {}           # job_id -> dict
_state_lock = threading.Lock()
_render_lock = threading.Lock()   # DSP is CPU-bound: one render at a time
_export_lock = threading.Lock()   # one full-song export at a time, beside the previews
_video_lock = threading.Lock()    # one Chrome render at a time
_yt_lock = threading.Lock()       # one YouTube upload at a time
_sources = {}        # source_id -> finished remix in output/ (tab 2)
_orig_cache = {}     # (track, a, b) -> (preview WAV of the original, lufs, peaks)
_audio_cache = {"path": None, "audio": None, "sr": None, "info": None, "stems": None, "restore": None}
_stem_support = {"checked": False, "available": False, "device": None}
# Short bursts only (true-peak chunks, file encode next to the metrics): half the threads, the rest stay free
_workers = ThreadPoolExecutor(max_workers=max(2, min(8, (os.cpu_count() or 4) // 2)), thread_name_prefix="studio")
_saver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-save")
TP_CHUNK = 441000   # true-peak chunk (samples)
TP_HALO = 64        # > the 4x resampler's reach (~11 input samples each side), so chunk interiors are exact


def stem_support() -> dict:
    """Importing torch takes seconds: check once (a background thread warms this at startup)."""
    if not _stem_support["checked"]:
        from core.stems import stems_available, stems_device
        available = stems_available()
        _stem_support.update(checked=True, available=available, device=stems_device() if available else None)
    return _stem_support


def track_stems(path: str, audio: np.ndarray, sr: int, progress) -> dict:
    """Stems of the loaded track: memory -> FLAC cache -> AI separation (first time ~30-60 s)."""
    if _audio_cache["path"] == path and _audio_cache["stems"] is not None:
        return _audio_cache["stems"]
    from core.stems import cached_stems
    stems = cached_stems(path, audio, sr, STEM_DIR, progress_callback=progress)
    if _audio_cache["path"] == path:
        _audio_cache["stems"] = stems
    return stems


def track_restore(path: str, audio: np.ndarray, sr: int, progress) -> Optional[np.ndarray]:
    """What the AI adds above the codec cutoff of the loaded track: memory -> FLAC cache -> model (~1 min)."""
    if _audio_cache["path"] == path and _audio_cache["restore"] is not None:
        return _audio_cache["restore"]
    from core.restore import cached_restore
    delta = cached_restore(path, audio, sr, RESTORE_DIR, progress_callback=progress)
    if _audio_cache["path"] == path:
        _audio_cache["restore"] = delta
    return delta


def _id_for(path: str) -> str:
    return hashlib.sha1(os.path.abspath(path).encode("utf-8")).hexdigest()[:12]


def register_media(path: str) -> str:
    media_id = _id_for(path) + os.path.splitext(path)[1].lower()
    with _state_lock:
        _media[media_id] = os.path.abspath(path)
    return f"/media/{media_id}"


def register_track(path: str) -> dict:
    track_id = _id_for(path)
    with _state_lock:
        _tracks[track_id] = os.path.abspath(path)
    return {"id": track_id, "name": os.path.basename(path),
            "source": "upload" if os.path.dirname(os.path.abspath(path)) == UPLOAD_DIR else "library"}


def scan_tracks() -> list:
    found = []
    for folder in (LIBRARY_DIR, UPLOAD_DIR):
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in AUDIO_EXTS:
                found.append(register_track(path))
    return found


# ---------------------------------------------------------------- audio helpers
def peaks(audio: np.ndarray, buckets: int = PEAK_BUCKETS) -> list:
    mono = np.max(np.abs(audio), axis=0)
    n = len(mono) // buckets
    if n < 1:
        return [round(float(v), 3) for v in mono]
    return [round(float(v), 3) for v in mono[:n * buckets].reshape(buckets, n).max(axis=1)]


def lufs(audio: np.ndarray, sr: int) -> float:
    try:
        value = integrated_loudness(audio, sr)
        return round(float(value), 2) if np.isfinite(value) else -70.0
    except ValueError:
        return -70.0


def true_peak_db(audio: np.ndarray) -> float:
    """Peak of resample_poly(audio, 4, 1) over the whole clip, computed on chunks with a halo in parallel: every
    kept sample is the same sum as in the one-shot version, so the maximum is identical, without its 1.2 GB
    float64 temporary on a full song (test_19)."""
    n = audio.shape[1]

    def chunk_peak(a: int) -> float:
        lo, hi = max(0, a - TP_HALO), min(n, a + TP_CHUNK + TP_HALO)
        up = signal.resample_poly(audio[:, lo:hi], 4, 1, axis=1)
        return float(np.max(np.abs(up[:, 4 * (a - lo): 4 * (min(n, a + TP_CHUNK) - lo)])))
    peak = max(_workers.map(chunk_peak, range(0, n, TP_CHUNK)), default=0.0)
    return round(20.0 * np.log10(peak + 1e-12), 2)


def loudest_window_start(audio: np.ndarray, sr: int, window_sec: float = PREVIEW_SEC) -> float:
    """Start of the loudest window (usually the chorus/drop) - the most telling part to audition."""
    mono = np.mean(audio, axis=0)
    blocks = len(mono) // sr
    win = int(window_sec)
    if blocks <= win:
        return 0.0
    energy = (mono[:blocks * sr].reshape(blocks, sr) ** 2).mean(axis=1)
    sums = np.convolve(energy, np.ones(win), mode="valid")
    return float(np.argmax(sums))


def load_track(path: str):
    """Decode once and keep the last track in memory (re-renders while tuning are instant)."""
    stat = os.stat(path)
    stamp = (stat.st_size, stat.st_mtime_ns)
    if _audio_cache["path"] != path or _audio_cache.get("stamp") != stamp:
        audio, sr = load_audio(path)
        _orig_cache.clear()
        _audio_cache.update(path=path, stamp=stamp, audio=audio, sr=sr, stems=None, restore=None, info={
            "duration": round(audio.shape[1] / sr, 2),
            "sample_rate": sr,
            "lufs": lufs(audio, sr),
            "peak": round(20.0 * np.log10(float(np.max(np.abs(audio))) + 1e-12), 2),
            "peaks": peaks(audio),
            "suggested_start": loudest_window_start(audio, sr),
            "source_cutoff": source_cutoff(audio, sr),   # Hz of the codec lowpass, None = full band
        })
    return _audio_cache["audio"], _audio_cache["sr"], _audio_cache["info"]


def unique_path(folder: str, stem: str, ext: str) -> str:
    path = os.path.join(folder, stem + ext)
    i = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{stem} ({i}){ext}")
        i += 1
    return path


def safe_name(name: str) -> str:
    name = os.path.basename(name).strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name)
    name = name.rstrip(". ") or "track"
    if name.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                        *(f"LPT{i}" for i in range(1, 10))}:
        name = "_" + name
    return name


def prune_previews(keep: int = 8):
    files = sorted(
        (os.path.join(PREVIEW_DIR, f) for f in os.listdir(PREVIEW_DIR)),
        key=os.path.getmtime, reverse=True,
    )
    for f in files[keep * 2:]:
        try:
            os.remove(f)
        except OSError:
            pass


# ---------------------------------------------------------------- jobs
class JobCancelled(Exception):
    pass


def run_render(job: dict, track_path: str, slug: str, knobs: dict, mode: str, start_sec: float, fmt: str,
               use_stems: bool = False, use_restore: bool = False):
    """Preview: under _render_lock, published as soon as both files and their loudness exist (state 'audio'),
    true-peak follows. Full export: only decoding/stems/restore take _render_lock, the DSP runs under _export_lock so
    previews keep working while a song exports in the background."""
    span = [0.1, 0.8]   # progress range of the current phase

    def progress(stage: str, frac: float):
        if job.get("cancel"):
            raise JobCancelled()
        job["stage"], job["progress"] = stage, round(span[0] + span[1] * frac, 3)

    def load():
        if job.get("cancel"):
            raise JobCancelled()
        job["state"], job["stage"] = "running", "Loading audio..."
        audio, sr, info = load_track(track_path)
        stems = None
        if use_stems:
            if not stem_support()["available"]:
                raise RuntimeError("Chế độ Stem cần cài: pip install demucs (và torch)")
            span[:] = [0.05, 0.55]
            stems = track_stems(track_path, audio, sr, progress)
            span[:] = [0.6, 0.3]
        delta = None
        if use_restore and info.get("source_cutoff"):   # a full-band source has nothing to restore
            from core.restore import restore_available
            if not restore_available():
                raise RuntimeError("Phục hồi dải cao cần model Apollo (xem README)")
            span[:] = [0.35, 0.25] if use_stems else [0.05, 0.55]
            delta = track_restore(track_path, audio, sr, progress)
            span[:] = [0.6, 0.3]
        return audio, sr, info, stems, delta

    try:
        if mode == "preview":
            with _render_lock:
                _render(job, track_path, slug, knobs, mode, start_sec, fmt, *load(), progress)
        else:
            job["stage"] = "Waiting for export..."
            with _export_lock:
                with _render_lock:
                    loaded = load()
                _render(job, track_path, slug, knobs, mode, start_sec, fmt, *loaded, progress)
    except JobCancelled:
        job["state"], job["error"] = "cancelled", "Đã hủy"
    except Exception as exc:  # surfaced to the UI
        job["state"], job["error"] = "error", f"{type(exc).__name__}: {exc}"


def preview_original(track_path: str, audio: np.ndarray, sr: int, a: int, b: int) -> tuple:
    """Original side of a preview (WAV + loudness + waveform), reused while tuning the same section."""
    key = (track_path, a, b)
    hit = _orig_cache.get(key)
    if hit and os.path.isfile(hit[0]):
        os.utime(hit[0])   # newest again, so prune_previews keeps it
        return hit, None
    seg = audio[:, a:b]
    path = os.path.join(PREVIEW_DIR, f"orig_{uuid.uuid4().hex[:10]}.wav")
    saving = _saver.submit(save_audio, path, seg, sr, subtype="PCM_16")
    hit = (path, lufs(seg, sr), peaks(seg))
    if len(_orig_cache) >= 8:
        _orig_cache.pop(next(iter(_orig_cache)))
    _orig_cache[key] = hit
    return hit, saving


def _render(job: dict, track_path: str, slug: str, knobs: dict, mode: str, start_sec: float, fmt: str,
            audio: np.ndarray, sr: int, info: dict, stems: Optional[dict],
            delta: Optional[np.ndarray], progress):
    preset = get_preset(slug)
    if mode == "preview":
        start_sec = float(np.clip(start_sec, 0.0, max(0.0, info["duration"] - 1.0)))
        a, b = int(start_sec * sr), int((start_sec + PREVIEW_SEC) * sr)
    else:
        start_sec = 0.0
        a, b = 0, audio.shape[1]
    seg = audio[:, a:b] if delta is None else audio[:, a:b] + delta[:, a:b]
    seg_stems = {name: stem[:, a:b] for name, stem in stems.items()} if stems else None

    t0 = time.perf_counter()
    remix = preset.apply(seg, sr, progress_callback=progress, knobs=knobs, stems=seg_stems)
    elapsed = time.perf_counter() - t0
    job["stage"], job["progress"] = "Saving...", 0.93

    base = os.path.splitext(os.path.basename(track_path))[0]
    # Files are written (MP3: ffmpeg, ~4 s per song) while the metrics below are measured
    orig_saving = None
    if mode == "preview":
        remix_path = os.path.join(PREVIEW_DIR, f"{job['id']}_remix.wav")

        def write():
            save_audio(remix_path, remix, sr, subtype="PCM_16")
            prune_previews()
        (orig_path, orig_lufs, orig_peaks), orig_saving = preview_original(track_path, audio, sr, a, b)
        saving = _saver.submit(write)
    else:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        remix_path = unique_path(OUTPUT_DIR, f"{base} ({slug})", "." + fmt)
        saving = _saver.submit(save_audio, remix_path, remix, sr, bitrate="320k")
        orig_path = track_path
        orig_lufs, orig_peaks = info["lufs"], info["peaks"]

    result = {
        "mode": mode,
        "start": start_sec,
        "speed": float(preset.build(knobs).speed),
        "orig_url": register_media(orig_path),
        "remix_url": register_media(remix_path),
        "orig_lufs": orig_lufs,
        "orig_peaks": orig_peaks,
        "remix_lufs": lufs(remix, sr),
        "remix_true_peak": None,
        "remix_peaks": peaks(remix),
        "remix_duration": round(remix.shape[1] / sr, 2),
        "elapsed": round(elapsed, 2),
        "rtf": round(elapsed / (remix.shape[1] / sr), 3),
        "output_path": remix_path if mode == "full" else None,
        "stems": bool(stems),
        "restore": delta is not None,
    }
    if mode == "preview":
        if orig_saving:
            orig_saving.result()
        saving.result()  # the files exist before the UI is told where they are
        job["result"], job["state"] = dict(result), "audio"   # playable now; true-peak follows
    result["remix_true_peak"] = true_peak_db(remix)
    saving.result()
    job["result"] = result
    job["state"], job["progress"], job["stage"] = "done", 1.0, "Done"


def new_job(stage: str = "Đang chờ...") -> dict:
    job = {"id": uuid.uuid4().hex[:10], "state": "queued", "progress": 0.0, "stage": stage, "result": None,
           "error": None, "cancel": False}
    _jobs[job["id"]] = job
    return job


# ---------------------------------------------------------------- tab 2: video & export
def write_json(path: str, data: dict):
    """Keep the previous project/brand readable if a write is interrupted."""
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)
    fd, partial = tempfile.mkstemp(suffix=".part", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)


def _brand() -> dict:
    from video import make
    return make.load_brand()


def scan_sources() -> list:
    """Finished remixes in output/ (newest first): what tab 2 turns into videos."""
    from video import make
    brand = _brand()
    out = []
    if os.path.isdir(OUTPUT_DIR):
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in REMIX_EXTS:
                out.append((os.path.getmtime(path), register_source(path, make.detect_style(path, brand))))
    return [s for _, s in sorted(out, key=lambda x: -x[0])]


def register_source(path: str, style: str = "") -> dict:
    sid = _id_for(path)
    with _state_lock:
        _sources[sid] = os.path.abspath(path)
    return {"id": sid, "name": os.path.basename(path), "style": style}


def _project_path(sid: str) -> str:
    return os.path.join(PROJECT_DIR, sid + ".json")


def source_info(sid: str, path: str) -> dict:
    """Everything tab 2 needs to open a remix: fields (with a cached AI answer if any, never a new agy call),
    saved project edits, audio URL, default art."""
    from video import make
    from video import shorts as SH
    from video import spectrum as SP
    brand = _brand()
    style = make.detect_style(path, brand)
    fields =make.gather(path, style, brand, None, None, "cache")
    spec = SP.cached(path)
    saved = None
    if os.path.isfile(_project_path(sid)):
        with open(_project_path(sid), encoding="utf-8") as f:
            saved = json.load(f)
    if saved and saved.get("lastResult"):
        saved["lastResult"] = export_links(saved["lastResult"])   # media ids live in memory: register again
    return {"id": sid, "name": os.path.basename(path), "style": style, "fields": fields, "brand": brand,
            "saved": saved, "audio_url": register_media(path), "duration": spec["dur"], "drop": spec["drop"],
            "bpm": spec["bpm"], "shorts": SH.pick_window(spec), "default_art": "/assets/" + os.path.basename(make.DEFAULT_IMAGE)
            if os.path.isfile(make.DEFAULT_IMAGE) else None}


def art_path(url: str):
    """UI art URL -> local file (uploaded art or the default logo), None when unknown."""
    if not url:
        return None
    rel = urllib.parse.unquote(urllib.parse.urlparse(url).path)
    if rel.startswith("/art/"):
        p = os.path.join(ART_DIR, os.path.basename(rel))
    else:
        p = os.path.realpath(os.path.join(SCRIPT_DIR, rel.lstrip("/")))
        if not os.path.normcase(p).startswith(os.path.normcase(os.path.realpath(os.path.join(SCRIPT_DIR, "assets"))) + os.sep):
            return None
    root = ART_DIR if rel.startswith("/art/") else os.path.join(SCRIPT_DIR, "assets")
    p = os.path.realpath(p)
    return p if (os.path.normcase(p).startswith(os.path.normcase(os.path.realpath(root)) + os.sep)
                 and os.path.splitext(p)[1].lower() in ART_EXTS and os.path.isfile(p)) else None


def run_ai(job: dict, path: str):
    from video import make
    try:
        job["state"], job["stage"], job["progress"] = "running", "AI đang tra tên bài (20–60 giây)...", 0.3
        brand = _brand()
        f = make.gather(path, make.detect_style(path, brand), brand, None, None, True)
        if f["ai"] and f["ai"].get("error"):
            raise RuntimeError(f["ai"]["error"])
        job["result"], job["state"], job["progress"], job["stage"] = f, "done", 1.0, "Xong"
    except Exception as exc:
        job["state"], job["error"] = "error", f"AI: {exc}"


def run_video(job: dict, path: str, d: dict):
    from video import make
    from video import render as R
    from video import spectrum as SP

    def stage(label, frac):
        job["stage"], job["progress"] = label, round(frac, 3)
    try:
        with _video_lock:
            if job["cancel"]:
                raise R.Cancelled("render cancelled")
            job["state"] = "running"
            stage("Tính phổ...", 0.01)
            brand = _brand()
            style = make.detect_style(path, brand)
            st = brand["styles"][style]
            image = art_path(d.get("art"))
            line1, hype = d.get("line1") or make.agy.clean_name(path), d.get("hype", "")
            spec = SP.cached(path)
            line2 = (d.get("line2") or "").strip() or (f"{st['name']} ✦ {hype}" if hype else st["name"])
            project = R.make_project(path, spec, line1, line2, st["name"],
                                     d.get("mood") or st["mood"], style, d.get("template"), image,
                                     focus=d.get("focus"), height=int(d.get("height", 1440)), handle=brand["handle"],
                                     hype=hype, fps=int(d.get("fps", R.DEFAULT_FPS)))
            project.update(look_options(d))
            name = make.folder_name({"ten_viet": line1, "phong_cach": st["name"]})
            res = make.produce(path, os.path.join(VIDEO_OUT_DIR, name), name, project, d.get("title", ""),
                               d.get("description", ""), image, d.get("goi_y") or [], brand, video=bool(d.get("video", True)),
                               shorts=bool(d.get("shorts")), seconds=d.get("seconds"), on_stage=stage, cancel=lambda: job["cancel"],
                               loop_minutes=60 if d.get("loop") and d.get("video", True) else None,
                               thumbnails=d.get("thumbnails", True))
            job["result"], job["state"], job["progress"], job["stage"] = export_links(res), "done", 1.0, "Xong"
    except Exception as exc:
        if job["cancel"]:
            job["state"], job["error"] = "cancelled", "Đã hủy"
        else:
            job["state"], job["error"] = "error", f"{type(exc).__name__}: {exc}"


LOOK_CLIPS = {"sad", "chill", "love", "dream", "hype", "cool"}
LOOK_FX = {"snow", "stars", "sakura", "sparks", "embers", "orbs", "none"}
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def look_options(d: dict) -> dict:
    """Tab 2 look overrides passed to template.js: chibi sequence, particles, accent colours (only known values)."""
    out = {}
    if d.get("clip") in LOOK_CLIPS:
        out["clip"] = d["clip"]
    if d.get("fx") in LOOK_FX:
        out["fx"] = d["fx"]
    if _HEX.match(str(d.get("a1", ""))) and _HEX.match(str(d.get("a2", ""))):
        out["a1"], out["a2"] = d["a1"], d["a2"]
    return out


def in_video_out(p: str, exts: set):
    """A file inside output/video/ with one of exts, else None (the UI only ever names files we exported)."""
    if not isinstance(p, str):
        return None
    p = os.path.realpath(p)
    ok = (os.path.normcase(p).startswith(os.path.normcase(os.path.realpath(VIDEO_OUT_DIR)) + os.sep)
          and os.path.splitext(p)[1].lower() in exts and os.path.isfile(p))
    return p if ok else None


def export_links(res):
    """Media URLs + YouTube upload records for an export result; files that disappeared are dropped."""
    from publish import youtube as YT
    if not isinstance(res, dict) or not isinstance(res.get("out_dir"), str):
        return None
    folder = os.path.realpath(res["out_dir"])
    root = os.path.realpath(VIDEO_OUT_DIR)
    if (not os.path.isdir(folder) or (os.path.normcase(folder) != os.path.normcase(root)
            and not os.path.normcase(folder).startswith(os.path.normcase(root) + os.sep))):
        return None
    for key in ("video", "shorts"):
        if isinstance(res.get(key), dict) and in_video_out(res[key].get("out"), {".mp4"}):
            res[key]["url"] = register_media(res[key]["out"])
            res[key]["youtube"] = YT.uploaded(res[key]["out"])
        else:
            res[key] = None
    thumbs = res.get("thumbnails") if isinstance(res.get("thumbnails"), list) else []
    res["thumbnails"] = [t for t in thumbs if isinstance(t, dict)
                         and in_video_out(t.get("path"), ART_EXTS)]
    for t in res["thumbnails"]:
        t["url"] = register_media(t["path"])
    return res


def run_yt_login(job: dict):
    from publish import youtube as YT
    try:
        job["state"], job["stage"] = "running", "Đang chờ anh đăng nhập Google trong trình duyệt..."
        job["result"] = YT.authorize(cancel=lambda: job["cancel"])
        job["state"], job["progress"], job["stage"] = "done", 1.0, "Xong"
    except YT.Cancelled:
        job["state"], job["error"] = "cancelled", "Đã hủy"
    except Exception as exc:
        job["state"], job["error"] = "error", str(exc)


def run_yt_upload(job: dict, src: str, video: str, thumb, d: dict):
    from publish import youtube as YT
    from video import make

    def prog(frac):
        job["stage"], job["progress"] = "Đang tải lên YouTube (Riêng tư)...", round(frac, 3)
    try:
        with _yt_lock:
            job["state"] = "running"
            prog(0.0)
            brand = _brand()
            style = brand["styles"][make.detect_style(src, brand)]["name"]
            f = d.get("fields") or {}
            tags = YT.tags_for(f.get("line1"), f.get("ten_goc"), f.get("ca_si"), f"{f.get('line1', '')} {style}",
                               style, brand.get("kenh")) if d.get("tags", True) else []
            job["result"] = YT.upload(video, d.get("title", ""), d.get("description", ""), tags, thumb,
                                      on_progress=prog, cancel=lambda: job["cancel"])
            job["state"], job["progress"], job["stage"] = "done", 1.0, "Xong"
    except YT.Cancelled:
        job["state"], job["error"] = "cancelled", "Đã hủy (bấm đăng lại sẽ tải tiếp từ chỗ dừng)"
    except Exception as exc:
        job["state"], job["error"] = "error", str(exc)


# ---------------------------------------------------------------- tính năng tải khi cần (MD91)
# Route cần một module có sẵn: thiếu module thì trả 404, không sập server. Khớp đúng hoặc theo tiền tố có "/" cuối.
_ROUTE_NEEDS = (
    ("/api/video/ai", ("video", "agy")),
    ("/api/brand", ("video",)),
    ("/api/video/", ("video",)),
    ("/api/youtube/", ("youtube",)),
    ("/api/agy/", ("agy",)),
)
INSTALLABLE = ("stems", "restore", "ffmpeg")


def module_presence(root: str = SCRIPT_DIR) -> dict:
    """Module riêng có mặt không. Bản public không kèm các module này nên tab và route tương ứng tự ẩn."""
    def co(*parts) -> bool:
        return os.path.isfile(os.path.join(root, *parts))
    return {"video": co("video", "__init__.py") and co("web", "video", "template.js"),
            "youtube": co("publish", "__init__.py"),
            "agy": co("ai", "__init__.py"),
            "ffmpeg": shutil.which("ffmpeg") is not None}


def route_present(path: str, present: dict) -> bool:
    for prefix, need in _ROUTE_NEEDS:
        if path == prefix or (prefix.endswith("/") and path.startswith(prefix)):
            return all(present[key] for key in need)
    return True


def features_payload() -> dict:
    """GET /api/features: module riêng có mặt + trạng thái tính năng tải khi cần + chữ cho hộp thoại."""
    return {**module_presence(), "stems": features.trang_thai("stems"), "restore": features.trang_thai("restore"),
            "gpu": features.gpu_co(), "mo_ta": features.mo_ta()}


def run_install(job: dict, ten: str, bien_the: str):
    """Cài trong nền. Tiến trình đi qua job như mọi tác vụ khác (GET /api/job/<id>)."""
    def tien(giai_doan: str, frac: float):
        job["stage"], job["progress"] = giai_doan, frac
    job["state"] = "running"
    try:
        if ten == "ffmpeg":
            features.cai_ffmpeg(tien)
        else:
            features.cai(ten, bien_the, tien)
        _stem_support.update(checked=False)  # lần đọc sau kiểm lại: torch + demucs vừa cài
        job["state"], job["progress"], job["stage"] = "done", 1.0, "Xong"
    except Exception as exc:
        job["state"], job["error"] = "error", str(exc) or type(exc).__name__


# ---------------------------------------------------------------- HTTP
class StudioHandler(BaseHTTPRequestHandler):
    server_version = "RemixStudio/2.0"

    def log_message(self, fmt, *args):
        pass  # keep the console readable

    # ------------------------------------------------ helpers
    def _local_request(self) -> bool:
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        host = self.headers.get("Host", "").lower()
        origin = self.headers.get("Origin")
        if (len(self.headers.get_all("Host", [])) != 1 or host not in hosts
                or (origin is not None and origin != "http://" + host)
                or self.headers.get("Sec-Fetch-Site") == "cross-site"):
            self.close_connection = True
            self._json({"error": "Chỉ nhận yêu cầu từ Meinya Studio trên máy này"}, 403)
            return False
        return True

    def _json(self, payload, status: int = 200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def _read_json(self) -> dict:
        data = json.loads(self._read_body(MAX_JSON_BYTES).decode("utf-8") or "{}")
        if not isinstance(data, dict):
            raise ValueError("JSON phải là object")
        json.dumps(data, allow_nan=False)  # reject NaN, Infinity and overflowing numbers anywhere in a project
        for key in ("track_id", "id", "preset", "mode", "format", "url", "path", "thumbnail", "title",
                    "description", "art", "line1", "line2", "hype", "mood", "template", "clip", "fx", "a1", "a2"):
            if key in data and ((key in ("title", "description") and data[key] is None)
                                or (data[key] is not None and not isinstance(data[key], str))):
                raise ValueError(f"{key} phải là chuỗi")
        for key in ("knobs", "project", "fields", "focus"):
            if key in data and data[key] is not None and not isinstance(data[key], dict):
                raise ValueError(f"{key} phải là object")
        for key in ("stems", "restore", "again", "video", "shorts", "loop", "tags", "thumbnails"):
            if key in data and not isinstance(data[key], bool):
                raise ValueError(f"{key} phải là boolean")
        for key in ("start", "seconds", "height", "fps"):
            if (key in data and data[key] is not None
                    and (isinstance(data[key], bool) or not isinstance(data[key], (int, float)))):
                raise ValueError(f"{key} phải là số")
        return data

    def _body_length(self, limit: int) -> int:
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) > 1:
            raise ValueError("Content-Length không hợp lệ")
        raw = lengths[0] if lengths else "0"
        if not re.fullmatch(r"[0-9]+", raw) or int(raw) > limit:
            raise ValueError("Nội dung rỗng, quá lớn hoặc Content-Length không hợp lệ")
        self.connection.settimeout(15)
        return int(raw)

    def _read_body(self, limit: int) -> bytes:
        length = self._body_length(limit)
        data = self.rfile.read(length)
        if len(data) != length:
            raise ValueError("Tải lên bị ngắt, chưa nhận đủ dữ liệu")
        return data

    def _serve_file(self, path: str):
        """Static/media file with HTTP Range support (required for seeking in <audio>)."""
        size = os.path.getsize(path)
        ctype = MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        match = re.match(r"bytes=(\d*)-(\d*)", rng)
        if match and (match.group(1) or match.group(2)):
            if match.group(1):
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
            else:
                start = max(0, size - int(match.group(2)))
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(256 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass  # browser cancelled (normal when seeking)

    # ------------------------------------------------ routes
    def do_GET(self):
        if not self._local_request():
            return
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/") and not route_present(path, module_presence()):
            return self._json({"error": "Tính năng này không có trong bản này"}, 404)
        if path == "/api/features":
            return self._json(features_payload())
        if path in ("/", "/index.html"):
            return self._serve_file(os.path.join(WEB_DIR, "index.html"))
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            if name in STATIC_FILES:
                return self._serve_file(os.path.join(WEB_DIR, name))
        if path.startswith("/media/"):
            with _state_lock:
                file_path = _media.get(path[len("/media/"):])
            if file_path and os.path.isfile(file_path):
                return self._serve_file(file_path)
        if path.startswith("/art/"):
            p = art_path(path)
            if p:
                return self._serve_file(p)
        top = urllib.parse.unquote(path).lstrip("/").split("/", 1)[0]
        if top in ROOT_SERVED:
            full = os.path.realpath(os.path.join(SCRIPT_DIR, urllib.parse.unquote(path).lstrip("/")))
            if (os.path.normcase(full).startswith(os.path.normcase(os.path.realpath(os.path.join(SCRIPT_DIR, top))) + os.sep)
                    and os.path.isfile(full)
                    and os.path.splitext(full)[1].lower() in ROOT_SERVED[top]):
                return self._serve_file(full)
        if path == "/api/brand":
            return self._json(_brand())
        if path == "/api/youtube/status":
            from publish import youtube as YT
            return self._json(YT.status())
        if path == "/api/agy/status":
            from ai import agy
            return self._json({"available": bool(agy.find_agy()), "model": agy.DEFAULT_MODEL})
        if path == "/api/video/sources":
            return self._json(scan_sources())
        if path.startswith("/api/video/source/") or path.startswith("/api/video/spectrum/"):
            sid = path.rsplit("/", 1)[-1]
            with _state_lock:
                src = _sources.get(sid)
            if not src or not os.path.isfile(src):
                return self._json({"error": "Không tìm thấy bản remix"}, 404)
            try:
                if "/spectrum/" in path:
                    from video import spectrum as SP
                    fps = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("fps", [""])[0]
                    return self._json(SP.cached(src, fps=int(fps) if fps in ("30", "60") else SP.FPS))
                return self._json(source_info(sid, src))
            except Exception as exc:
                return self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)
        if path == "/api/presets":
            return self._json([
                {"slug": p.slug, "name": p.name, "description": p.description,
                 "knobs": p.build().get_knobs()}
                for p in list_presets()
            ])
        if path == "/api/tracks":
            return self._json(scan_tracks())
        if path.startswith("/api/track/"):
            with _state_lock:
                track_path = _tracks.get(path[len("/api/track/"):])
            if not track_path:
                return self._json({"error": "Không tìm thấy bài hát"}, 404)
            try:
                with _render_lock:
                    _, sr, info = load_track(track_path)
                from core.stems import stems_cached
                from core.restore import restore_available, restore_cached
                support = stem_support()
                return self._json({**info, "stems_available": support["available"],
                                   "stems_device": support["device"],
                                   "stems_ready": stems_cached(track_path, sr, STEM_DIR),
                                   "restore_available": restore_available(),
                                   "restore_ready": restore_cached(track_path, sr, RESTORE_DIR)})
            except Exception as exc:
                return self._json({"error": f"Không đọc được file: {exc}"}, 400)
        if path.startswith("/api/job/"):
            job = _jobs.get(path[len("/api/job/"):])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            return self._json({k: v for k, v in job.items() if k != "id"})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if not self._local_request():
            return
        try:
            self._post()
        except (ValueError, TypeError, OverflowError, TimeoutError) as exc:
            self.close_connection = True
            self._json({"error": f"Dữ liệu không hợp lệ: {exc}"}, 400)
        except OSError:
            self.close_connection = True
            self._json({"error": "Không đọc/ghi được file. Kiểm tra dung lượng và quyền thư mục."}, 500)

    def _post(self):
        path = urllib.parse.urlparse(self.path).path
        if not route_present(path, module_presence()):
            self.close_connection = True  # body chưa đọc: đóng kết nối để không lệch yêu cầu sau
            return self._json({"error": "Tính năng này không có trong bản này"}, 404)
        if path.startswith("/api/features/") and path.endswith("/install"):
            ten = path[len("/api/features/"):-len("/install")]
            if ten not in INSTALLABLE:
                return self._json({"error": "Không có tính năng này"}, 404)
            bien_the = self._read_json().get("bien_the") or "cpu"
            if bien_the not in ("cpu", "cuda"):
                return self._json({"error": "bien_the phải là cpu hoặc cuda"}, 400)
            if bien_the == "cuda" and not features.gpu_co():
                return self._json({"error": "Máy không thấy card NVIDIA (nvidia-smi), hãy chọn bản CPU"}, 400)
            da_co = features.ffmpeg_co() if ten == "ffmpeg" else features.trang_thai(ten)["co"]
            if da_co:
                return self._json({"error": "Tính năng này đã có sẵn"}, 409)
            job = new_job("Đang chờ...")
            threading.Thread(target=run_install, args=(job, ten, bien_the), daemon=True).start()
            return self._json({"job_id": job["id"]})

        if path == "/api/upload":
            length = self._body_length(MAX_UPLOAD_BYTES)
            if length <= 0 or length > MAX_UPLOAD_BYTES:
                return self._json({"error": "File rỗng hoặc quá lớn (tối đa 500 MB)"}, 400)
            name = safe_name(urllib.parse.unquote(self.headers.get("X-Filename", "track")))
            if os.path.splitext(name)[1].lower() not in AUDIO_EXTS:
                return self._json({"error": "Chỉ nhận MP3, WAV, FLAC, M4A, OGG, AAC, OPUS"}, 400)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            fd, partial = tempfile.mkstemp(suffix=".part", dir=UPLOAD_DIR)
            try:
                with os.fdopen(fd, "wb") as f:
                    remaining = length
                    while remaining > 0:
                        chunk = self.rfile.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError("Tải lên bị ngắt, chưa nhận đủ dữ liệu")
                        f.write(chunk)
                        remaining -= len(chunk)
                with _state_lock:
                    stem, ext = os.path.splitext(name)
                    dest = unique_path(UPLOAD_DIR, stem, ext)
                    os.replace(partial, dest)
            finally:
                if os.path.exists(partial):
                    os.remove(partial)
            return self._json(register_track(dest))

        if path == "/api/pin":
            # Copy a finished remix so it survives preview pruning (reference for blind A/B)
            data = self._read_json()
            with _state_lock:
                src = _media.get((data.get("url") or "").rsplit("/", 1)[-1])
            if not src or not os.path.isfile(src):
                return self._json({"error": "Không tìm thấy bản cần ghim"}, 404)
            os.makedirs(PINNED_DIR, exist_ok=True)
            dest = os.path.join(PINNED_DIR, f"{uuid.uuid4().hex[:10]}{os.path.splitext(src)[1]}")
            shutil.copyfile(src, dest)
            return self._json({"url": register_media(dest)})

        if path == "/api/render":
            data = self._read_json()
            with _state_lock:
                track_path = _tracks.get(data.get("track_id", ""))
            slug = data.get("preset", "")
            mode = data.get("mode", "preview")
            fmt = data.get("format", "wav")
            if not track_path:
                return self._json({"error": "Chưa chọn bài hát"}, 400)
            if get_preset(slug) is None:
                return self._json({"error": f"Preset không tồn tại: {slug}"}, 400)
            if mode not in ("preview", "full") or fmt not in ("wav", "mp3", "flac"):
                return self._json({"error": "mode/format không hợp lệ"}, 400)
            knobs = data.get("knobs") or {}
            get_preset(slug).build(knobs)  # validate before allocating a background job
            start = float(data.get("start", 0.0))
            if not np.isfinite(start) or start < 0:
                raise ValueError("start phải là số hữu hạn không âm")
            job = new_job()
            threading.Thread(
                target=run_render,
                args=(job, track_path, slug, knobs, mode, start, fmt, bool(data.get("stems")),
                      bool(data.get("restore"))),
                daemon=True,
            ).start()
            return self._json({"job_id": job["id"]})

        if path.startswith("/api/job/") and path.endswith("/cancel"):
            job = _jobs.get(path.split("/")[3])
            if not job:
                return self._json({"error": "unknown job"}, 404)
            job["cancel"] = True
            return self._json({"ok": True})

        if path == "/api/video/art":
            length = self._body_length(50 * 1024 * 1024)
            ext = os.path.splitext(safe_name(urllib.parse.unquote(self.headers.get("X-Filename", "art.jpg"))))[1].lower()
            if length <= 0 or length > 50 * 1024 * 1024 or ext not in ART_EXTS:
                return self._json({"error": "Chỉ nhận ảnh PNG/JPG/WEBP tối đa 50 MB"}, 400)
            data = self._read_body(50 * 1024 * 1024)
            os.makedirs(ART_DIR, exist_ok=True)
            name = hashlib.sha1(data).hexdigest()[:16] + ext
            with open(os.path.join(ART_DIR, name), "wb") as f:
                f.write(data)
            return self._json({"url": "/art/" + name})

        if path == "/api/brand":
            d = self._read_json()
            from video import make
            brand = _brand()
            for k in ("kenh", "handle", "link_kenh", "email", "tieu_de", "mo_ta"):
                if isinstance(d.get(k), str):
                    brand[k] = d[k]
            write_json(make.BRAND_PATH, brand)
            return self._json(brand)

        if path in ("/api/video/ai", "/api/video/project", "/api/video/render"):
            d = self._read_json()
            with _state_lock:
                src = _sources.get(d.get("id", ""))
            if not src or not os.path.isfile(src):
                return self._json({"error": "Chưa chọn bản remix"}, 400)
            if path == "/api/video/project":
                os.makedirs(PROJECT_DIR, exist_ok=True)
                write_json(_project_path(d["id"]), d.get("project") or {})
                return self._json({"ok": True})
            if path == "/api/video/render" and len(d.get("title", "")) > 100:
                return self._json({"error": "Tiêu đề dài quá 100 ký tự (YouTube chặn)"}, 400)
            if path == "/api/video/render":
                from video import render as R
                R.make_project(src, {}, "", template=d.get("template"), focus=d.get("focus"),
                               height=d.get("height", 1440), fps=d.get("fps", R.DEFAULT_FPS))
                if d.get("seconds") is not None and d["seconds"] <= 0:
                    raise ValueError("seconds phải là số dương")
            job = new_job()
            target = (run_ai, (job, src)) if path == "/api/video/ai" else (run_video, (job, src, d))
            threading.Thread(target=target[0], args=target[1], daemon=True).start()
            return self._json({"job_id": job["id"]})

        if path == "/api/youtube/login":
            from publish import youtube as YT
            if not YT.find_client():
                return self._json({"error": "Chưa có OAuth client: đặt file JSON 'Desktop app' vào thư mục secret/"}, 400)
            job = new_job()
            threading.Thread(target=run_yt_login, args=(job,), daemon=True).start()
            return self._json({"job_id": job["id"]})

        if path == "/api/youtube/logout":
            from publish import youtube as YT
            YT.logout()
            return self._json(YT.status())

        if path == "/api/youtube/upload":
            # Always Private (D7): nothing in the request can change that, publish.youtube hard-codes it
            from publish import youtube as YT
            d = self._read_json()
            with _state_lock:
                src = _sources.get(d.get("id", ""))
            video = in_video_out(d.get("path"), {".mp4"})
            thumb = in_video_out(d.get("thumbnail"), {".jpg", ".jpeg", ".png"}) if d.get("thumbnail") else None
            if not src or not video:
                return self._json({"error": "Không tìm thấy video đã xuất"}, 400)
            errs = YT.check_meta(d.get("title", ""), d.get("description", ""))
            if errs:
                return self._json({"error": "Chưa đăng được: " + "; ".join(errs)}, 400)
            if not YT.status()["authorized"]:
                return self._json({"error": "Chưa đăng nhập kênh YouTube"}, 400)
            done = YT.uploaded(video)
            if done and not d.get("again"):
                return self._json({"error": f"Video này đã đăng lúc {done['at']}: {done['url']}", "uploaded": done}, 409)
            job = new_job()
            threading.Thread(target=run_yt_upload, args=(job, src, video, thumb, d), daemon=True).start()
            return self._json({"job_id": job["id"]})

        if path == "/api/open":
            p = os.path.realpath(self._read_json().get("path") or "")
            if (not os.path.normcase(p).startswith(os.path.normcase(os.path.realpath(OUTPUT_DIR)) + os.sep)
                    or not os.path.isdir(p) or not hasattr(os, "startfile")):
                return self._json({"error": "Chỉ mở được thư mục trong output/"}, 400)
            os.startfile(p)
            return self._json({"ok": True})

        self._json({"error": "not found"}, 404)


def run_web_studio(port: int = 0, open_browser: bool = True):
    if sys.platform != "win32":
        return _serve_web_studio(port, open_browser)
    import msvcrt
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, "studio.lock"), "a+b") as lock:
        if os.fstat(lock.fileno()).st_size == 0:
            lock.write(b" ")
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print("[!] Meinya Studio đang chạy. Dùng cửa sổ đã mở; tác vụ hiện tại vẫn tiếp tục.")
            return
        _serve_web_studio(port, open_browser)


def _serve_web_studio(port: int, open_browser: bool):
    for folder in (OUTPUT_DIR, UPLOAD_DIR, PREVIEW_DIR, PINNED_DIR, STEM_DIR, RESTORE_DIR):
        os.makedirs(folder, exist_ok=True)
    for folder in (PREVIEW_DIR, PINNED_DIR):
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except OSError:
                pass
    threading.Thread(target=stem_support, daemon=True).start()

    server = None
    for candidate in ([0] if port == 0 else range(port, port + 20)):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), StudioHandler)
            port = server.server_port
            break
        except OSError:
            continue
    if server is None:
        raise SystemExit(f"[!] No free port in {port}-{port + 19}")

    url = f"http://127.0.0.1:{port}"
    print("=" * 64)
    print("  REMIX STUDIO - local web UI")
    print(f"  URL    : {url}")
    print(f"  Output : {OUTPUT_DIR}")
    print("  Ctrl+C to stop.")
    print("=" * 64)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
        # Only this session's temporary files; never another studio's active previews.
        for folder in (PREVIEW_DIR, PINNED_DIR):
            try:
                for name in os.listdir(folder):
                    os.remove(os.path.join(folder, name))
                os.rmdir(folder)
            except OSError:
                pass


if __name__ == "__main__":
    if sys.platform.startswith("win"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except AttributeError:
            pass
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run_web_studio(int(args[0]) if args else 0, open_browser="--no-browser" not in sys.argv)
