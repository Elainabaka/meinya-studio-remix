"""Tính năng tải khi cần (MD91): Stem AI, bù dải cao AI (Apollo) và ffmpeg.

Bản public không kèm torch, Demucs, mã Apollo hay trọng số. Người dùng bấm "Tải về" thì mới tải, và chỉ tải
đúng gói đã ghim từ nguồn chính thức. File tải về ghi ra file tạm, kiểm SHA-256, khớp mới đổi tên; không chạy
code tải về trước khi kiểm.
"""

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Phiên bản đang chạy được trên máy chủ nhân (py -3 -m pip show torch demucs omegaconf huggingface_hub)
TORCH = "2.13.0"
TORCH_INDEX = {"cpu": ("+cpu", "https://download.pytorch.org/whl/cpu"),
               "cuda": ("+cu126", "https://download.pytorch.org/whl/cu126")}

# Apollo: mã lấy từ github.com/JusperLee/Apollo ở một commit ghim, SHA-256 từng file đã kiểm trên commit đó
APOLLO_COMMIT = "e84bcacc59d5455f05d86a5c97dd4aeb3c14dbb6"
APOLLO_RAW = f"https://raw.githubusercontent.com/JusperLee/Apollo/{APOLLO_COMMIT}"
APOLLO_DIR = os.path.join(ROOT, "core", "apollo")
WEIGHTS_DIR = os.path.join(ROOT, ".studio_cache", "models")
WEIGHTS_URL = "https://huggingface.co/JusperLee/Apollo/resolve/main/pytorch_model.bin"
WEIGHTS_SHA256 = "99d9af7f1ff20e63c393035513a655392818d66b4d7fc23d658175c1f15e8d76"

FEATURES = {
    "stems": {
        "ten": "Stem AI",
        "mo_ta": "Tách bài thành giọng, trống, bass và nhạc cụ để xử lý riêng từng phần.",
        "torch": True,
        "goi": ["demucs==4.1.0", "einops==0.8.2", "huggingface-hub==1.14.0", "julius==0.2.8", "lameenc==1.8.4",
                "pyyaml==6.0.3", "safetensors==0.8.0", "sphn==0.2.1", "tqdm==4.67.3"],
        "tep": [],
        "dung_luong": "torch: khoảng 2,6 GB bản GPU (CUDA 12.6) hoặc 0,12 GB bản CPU (đo từ index PyTorch); "
                      "model Demucs khoảng 300 MB tải thêm lần đầu chạy (ước tính)",
        "nguon": "download.pytorch.org (torch) và PyPI (demucs); model htdemucs_ft do Demucs tải lần đầu",
        "giay_phep": "PyTorch: BSD; Demucs: MIT; model htdemucs_ft: theo repo Demucs (MIT)",
    },
    "restore": {
        "ten": "Bù dải cao AI (Apollo)",
        "mo_ta": "Bù phần dải cao mà file MP3 đã cắt (thường trên 16 kHz). Chỉ phần bị cắt được thay, "
                 "phần còn lại của bài giữ nguyên.",
        "torch": True,
        "goi": ["omegaconf==2.3.1", "huggingface_hub==1.14.0"],
        "init_apollo": True,
        "tep": [
            {"url": f"{APOLLO_RAW}/LICENSE", "sha256": "eb8364a4474d5b6e207f3800c75b04070dde1f4ea9c210c38cd836f7a3f24203",
             "dich": os.path.join(APOLLO_DIR, "LICENSE")},
            {"url": f"{APOLLO_RAW}/look2hear/models/apollo.py",
             "sha256": "e34d96c2f0320f34f1297023652b741b7e5e1028fe12a946cbf9faf95f442e89",
             "dich": os.path.join(APOLLO_DIR, "apollo.py")},
            {"url": f"{APOLLO_RAW}/look2hear/models/base_model.py",
             "sha256": "8fc90f5af762ae1d0d26cc51ec03ef4a2f956483d63d72e0d39b040f90e7c598",
             "dich": os.path.join(APOLLO_DIR, "base_model.py")},
            {"url": WEIGHTS_URL, "sha256": WEIGHTS_SHA256, "dich": os.path.join(WEIGHTS_DIR, "apollo.bin")},
        ],
        "dung_luong": "torch như trên, cộng khoảng 66 MB trọng số Apollo (đo trên máy chủ nhân) và vài chục KB mã",
        "nguon": "Mã: github.com/JusperLee/Apollo (commit ghim). Trọng số: huggingface.co/JusperLee/Apollo. "
                 "Mọi file kiểm SHA-256",
        "giay_phep": "Mã Apollo và trọng số: CC BY-SA 4.0 (LICENSE của repo gốc; trang model ghi cc-by-sa-4.0); "
                     "omegaconf: BSD-3-Clause; huggingface_hub: Apache-2.0",
    },
}

FFMPEG = {
    "ten": "ffmpeg",
    "mo_ta": "Xuất MP3 và đọc các định dạng mà thư viện âm thanh không đọc được. Không có thì vẫn xuất WAV, FLAC, OGG.",
    "winget": "Gyan.FFmpeg",
    "ban": "8.0.1",   # bản đang chạy trên máy chủ nhân; ghim để mọi máy cài đúng bản đã kiểm
    "trang": "https://ffmpeg.org/download.html",
    "dung_luong": "khoảng 234 MB tải về, khoảng 616 MB sau khi cài (đo từ bản 8.0.1)",
    "nguon": "winget, gói Gyan.FFmpeg 8.0.1 (zip từ github.com/GyanD/codexffmpeg, winget kiểm SHA-256). Trang chính thức: ffmpeg.org",
    "giay_phep": "GPL-3.0 (bản build full của Gyan; xem ffmpeg.org/legal.html)",
}

_khoa = threading.Lock()
_dang_cai = set()
_loi = {}


def _tinh_nang(ten: str) -> dict:
    if ten not in FEATURES:
        raise ValueError(f"Không có tính năng: {ten}")
    return FEATURES[ten]


def _pip_mac_dinh(args: list):
    """pip của chính Python đang chạy app (trong .venv khi chạy bằng run.bat), không cài vào Python hệ thống."""
    ket_qua = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", *args],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
    if ket_qua.returncode != 0:
        duoi = (ket_qua.stderr or ket_qua.stdout).strip().splitlines()[-3:]
        raise RuntimeError("pip lỗi: " + " | ".join(duoi))


def tai_file(url: str, dich: str, sha256: str, tien=None, mo=None):
    """Tải về file tạm trong cùng thư mục, kiểm SHA-256, khớp mới đổi tên vào đích. Sai thì xóa file tạm, báo lỗi."""
    thu_muc = os.path.dirname(dich)
    os.makedirs(thu_muc, exist_ok=True)
    fd, tam = tempfile.mkstemp(dir=thu_muc, suffix=".part")
    os.close(fd)
    try:
        bam = hashlib.sha256()
        with (mo or urllib.request.urlopen)(url, timeout=60) as nguon, open(tam, "wb") as f:
            tong = int(nguon.headers.get("Content-Length") or 0)
            da_tai = 0
            while chunk := nguon.read(1 << 20):
                f.write(chunk)
                bam.update(chunk)
                da_tai += len(chunk)
                if tien and tong:
                    tien(min(1.0, da_tai / tong))
        if bam.hexdigest() != sha256.lower():
            raise ValueError(f"SHA-256 của {os.path.basename(dich)} không khớp: file tải về bị hỏng hoặc không phải "
                             "bản đã kiểm. Đã bỏ file, thử lại sau.")
        os.replace(tam, dich)
    except BaseException:
        try:
            os.remove(tam)
        except OSError:
            pass
        raise


def _co(ten: str) -> bool:
    if ten == "stems":
        try:
            return all(importlib.util.find_spec(m) is not None for m in ("torch", "demucs"))
        except (ImportError, ValueError):
            return False
    from core.restore import restore_available
    return restore_available()


def gpu_co() -> bool:
    """Có nvidia-smi: để hỏi người dùng chọn bản CUDA (nhanh, nặng) hay CPU (nhẹ, chậm)."""
    return shutil.which("nvidia-smi") is not None


def trang_thai(ten: str) -> dict:
    _tinh_nang(ten)
    with _khoa:
        dang = ten in _dang_cai
        loi = _loi.get(ten)
    return {"co": _co(ten), "dang_cai": dang, "loi": loi, "gpu": gpu_co()}


def mo_ta() -> dict:
    """Chữ hiển thị trong hộp thoại "Tải về": tên, nội dung, dung lượng, nguồn, giấy phép."""
    chi_tiet = {}
    for ten, f in FEATURES.items():
        chi_tiet[ten] = {k: f[k] for k in ("ten", "mo_ta", "dung_luong", "nguon", "giay_phep")}
    chi_tiet["ffmpeg"] = {k: FFMPEG[k] for k in ("ten", "mo_ta", "dung_luong", "nguon", "giay_phep")}
    return chi_tiet


def cai(ten: str, bien_the: str = "cpu", tien_trinh=None, pip=None, tai=None):
    """Cài một tính năng đã ghim. bien_the: "cpu" hoặc "cuda" (chỉ đổi bản torch). Lỗi được ghi vào trang_thai."""
    f = _tinh_nang(ten)
    if bien_the not in TORCH_INDEX:
        raise ValueError(f"bien_the phải là cpu hoặc cuda, không phải {bien_the}")
    with _khoa:
        if ten in _dang_cai:
            raise RuntimeError("Đang cài tính năng này, đợi xong đã")
        _dang_cai.add(ten)
        _loi.pop(ten, None)
    tien = tien_trinh or (lambda giai_doan, frac: None)
    pip = pip or _pip_mac_dinh
    tai = tai or tai_file
    try:
        if f["torch"]:
            tag, index = TORCH_INDEX[bien_the]
            tien("Cài thư viện đã ghim", 0.0)
            pip(["--extra-index-url", index, f"torch=={TORCH}{tag}", *f["goi"]])
        else:
            tien("Cài thư viện đã ghim", 0.0)
            pip(list(f["goi"]))
        tep = f["tep"]
        for i, t in enumerate(tep):
            ten_file = os.path.basename(t["dich"])
            tai(t["url"], t["dich"], t["sha256"],
                lambda frac, i=i, ten_file=ten_file: tien(f"Tải {ten_file}", (i + frac) / len(tep)))
        if f.get("init_apollo"):
            # core.restore nạp mã bằng "from .apollo.apollo import Apollo": cần package, file rỗng không tải từ mạng
            os.makedirs(APOLLO_DIR, exist_ok=True)
            open(os.path.join(APOLLO_DIR, "__init__.py"), "a", encoding="utf-8").close()
        tien("Xong", 1.0)
    except BaseException as exc:
        with _khoa:
            _loi[ten] = str(exc) or type(exc).__name__
        raise
    finally:
        with _khoa:
            _dang_cai.discard(ten)


def ffmpeg_co() -> bool:
    return shutil.which("ffmpeg") is not None


def _winget_mac_dinh(args: list):
    ket_qua = subprocess.run(["winget", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if ket_qua.returncode != 0:
        duoi = (ket_qua.stdout or ket_qua.stderr).strip().splitlines()[-3:]
        raise RuntimeError("winget lỗi: " + " | ".join(duoi))


def cai_ffmpeg(tien_trinh=None, chay=None):
    """Cài ffmpeg bằng winget (gói chính thức Gyan.FFmpeg). Không có winget thì báo đường dẫn tải tay."""
    tien = tien_trinh or (lambda giai_doan, frac: None)
    if chay is None and not shutil.which("winget"):
        raise RuntimeError(f"Máy chưa có winget. Tải ffmpeg tại {FFMPEG['trang']}, rồi thêm vào PATH.")
    tien("winget đang cài ffmpeg (có thể vài phút)", 0.1)
    (chay or _winget_mac_dinh)(["install", "-e", "--id", FFMPEG["winget"], "--version", FFMPEG["ban"],
                                "--accept-package-agreements", "--accept-source-agreements"])
    # winget đặt ffmpeg sau shim trong WinGet\Links; app đang chạy chưa có trong PATH nên thêm vào tại đây
    links = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Links")
    if os.path.isdir(links) and links not in os.environ.get("PATH", ""):
        os.environ["PATH"] = links + os.pathsep + os.environ.get("PATH", "")
    if not ffmpeg_co():
        raise RuntimeError("Cài xong nhưng chưa thấy ffmpeg. Đóng app, mở lại rồi bấm cài lại.")
    tien("Xong", 1.0)
