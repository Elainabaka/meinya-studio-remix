"""Tính năng tải khi cần (core/features.py): tải kiểm SHA-256, ghim phiên bản, lỗi không làm hỏng app.

Không gọi mạng và không chạy pip thật: tải và pip được thay bằng bản giả.
"""
import hashlib
import io
import os
import tempfile
import unittest
from unittest import mock

from core import features as F
from core import restore as R


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}


class TaiFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "sub")
        self.dest = os.path.join(self.dir, "model.bin")
        self.data = b"trong so gia"

    def tearDown(self):
        self.tmp.cleanup()

    def test_sai_sha_thi_bao_loi_va_khong_de_file_nao(self):
        with mock.patch.object(F.urllib.request, "urlopen", return_value=FakeResponse(self.data)):
            with self.assertRaises(ValueError):
                F.tai_file("https://example.invalid/x", self.dest, "0" * 64)
        self.assertFalse(os.path.exists(self.dest))
        self.assertEqual(os.listdir(self.dir), [])  # file tạm cũng đã bị xóa

    def test_dung_sha_thi_dat_dung_cho_va_khong_con_file_tam(self):
        sha = hashlib.sha256(self.data).hexdigest()
        with mock.patch.object(F.urllib.request, "urlopen", return_value=FakeResponse(self.data)):
            F.tai_file("https://example.invalid/x", self.dest, sha)
        with open(self.dest, "rb") as f:
            self.assertEqual(f.read(), self.data)
        self.assertEqual(os.listdir(self.dir), ["model.bin"])

    def test_loi_mang_giua_chung_khong_de_file_do_dang_do(self):
        class Dut(FakeResponse):
            def read(self, n=-1):
                raise ConnectionResetError("đứt mạng")
        with mock.patch.object(F.urllib.request, "urlopen", return_value=Dut(self.data)):
            with self.assertRaises(ConnectionResetError):
                F.tai_file("https://example.invalid/x", self.dest, "0" * 64)
        self.assertEqual(os.listdir(self.dir), [])


class CaiTest(unittest.TestCase):
    def setUp(self):
        self.pip_calls = []
        self.tai_calls = []

    def pip_ok(self, args):
        self.pip_calls.append(list(args))

    def tai_ok(self, url, dich, sha, tien=None):
        self.tai_calls.append((url, dich, sha))

    def test_stems_ghim_torch_cpu_tu_index_chinh_thuc(self):
        F.cai("stems", "cpu", pip=self.pip_ok, tai=self.tai_ok)
        args = self.pip_calls[0]
        self.assertEqual(args[:3], ["--extra-index-url", "https://download.pytorch.org/whl/cpu", "torch==2.13.0+cpu"])
        self.assertIn("demucs==4.1.0", args)
        self.assertEqual(self.tai_calls, [])

    def test_stems_ban_gpu_dung_cu126(self):
        F.cai("stems", "cuda", pip=self.pip_ok, tai=self.tai_ok)
        args = self.pip_calls[0]
        self.assertEqual(args[:3], ["--extra-index-url", "https://download.pytorch.org/whl/cu126", "torch==2.13.0+cu126"])

    def test_bien_the_la_bi_tu_choi(self):
        with self.assertRaises(ValueError):
            F.cai("stems", "gpu", pip=self.pip_ok, tai=self.tai_ok)
        self.assertEqual(self.pip_calls, [])

    def test_restore_tai_du_bon_file_ghim_commit_va_sha(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(F, "APOLLO_DIR", os.path.join(tmp, "apollo")), \
                mock.patch.object(F, "WEIGHTS_DIR", os.path.join(tmp, "models")):
            F.cai("restore", "cpu", pip=self.pip_ok, tai=self.tai_ok)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "apollo", "__init__.py")))
        self.assertEqual(len(self.tai_calls), 4)
        for url, dich, sha in self.tai_calls:
            if "huggingface" in url:
                self.assertEqual(url, F.WEIGHTS_URL)
            else:
                self.assertIn(F.APOLLO_COMMIT, url)
            self.assertEqual(len(sha), 64)
        names = sorted(os.path.basename(dich) for _, dich, _ in self.tai_calls)
        self.assertEqual(names, ["LICENSE", "apollo.bin", "apollo.py", "base_model.py"])

    def test_pin_khop_voi_core_restore(self):
        # Hai nơi cùng giữ mã Apollo và trọng số: lệch nhau thì bản cài và bản chạy không khớp
        self.assertEqual(F.WEIGHTS_SHA256, R.WEIGHTS_SHA256)
        self.assertEqual(F.WEIGHTS_URL, R.WEIGHTS_URL)

    def test_pip_loi_ghi_loi_va_khong_ket_tinh_nang(self):
        def pip_fail(args):
            raise RuntimeError("pip lỗi: mạng đứt")
        with self.assertRaises(RuntimeError):
            F.cai("stems", "cpu", pip=pip_fail, tai=self.tai_ok)
        st = F.trang_thai("stems")
        self.assertIn("pip lỗi", st["loi"])
        self.assertFalse(st["dang_cai"])
        # Lần sau cài được thì lỗi cũ bị xóa
        F.cai("stems", "cpu", pip=self.pip_ok, tai=self.tai_ok)
        self.assertIsNone(F.trang_thai("stems")["loi"])

    def test_ten_la_bi_tu_choi(self):
        with self.assertRaises(ValueError):
            F.trang_thai("khong_co")

    def test_ffmpeg_cai_bang_winget_tu_nguon_chinh_thuc(self):
        calls = []
        with mock.patch.object(F, "ffmpeg_co", return_value=True):
            F.cai_ffmpeg(chay=lambda args: calls.append(list(args)))
        self.assertEqual(calls[0][:4], ["install", "-e", "--id", "Gyan.FFmpeg"])

    def test_ffmpeg_cai_xong_ma_khong_thay_thi_bao_loi(self):
        with mock.patch.object(F, "ffmpeg_co", return_value=False):
            with self.assertRaises(RuntimeError):
                F.cai_ffmpeg(chay=lambda args: None)

    def test_hop_thoai_co_du_chu_cho_moi_tinh_nang(self):
        mo = F.mo_ta()
        for ten in ("stems", "restore", "ffmpeg"):
            for key in ("ten", "mo_ta", "dung_luong", "nguon", "giay_phep"):
                self.assertTrue(mo[ten][key], f"{ten}.{key} trống")


class TrangThaiTest(unittest.TestCase):
    def test_restore_co_khi_du_file_va_thu_vien(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = os.path.join(tmp, "apollo")
            os.makedirs(model_dir)
            weights = os.path.join(tmp, "apollo.bin")
            for name in ("apollo.py", "base_model.py"):
                open(os.path.join(model_dir, name), "w").close()
            open(weights, "wb").close()
            with mock.patch.object(R, "MODEL_DIR", model_dir), mock.patch.object(R, "WEIGHTS", weights), \
                    mock.patch("importlib.util.find_spec", return_value=object()):
                self.assertTrue(F.trang_thai("restore")["co"])
                os.remove(weights)
                self.assertFalse(F.trang_thai("restore")["co"])

    def test_gpu_theo_nvidia_smi(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertFalse(F.gpu_co())
        with mock.patch("shutil.which", return_value="C:/nvidia-smi.exe"):
            self.assertTrue(F.gpu_co())


if __name__ == "__main__":
    unittest.main()
