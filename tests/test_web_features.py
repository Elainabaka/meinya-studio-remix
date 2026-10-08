"""API tính năng tải khi cần trên server thật (cổng tạm, 127.0.0.1).

Thiếu module có sẵn: route trả 404, server vẫn chạy. Cài lỗi: job báo lỗi, server vẫn chạy.
Không gọi mạng, không chạy pip thật.
"""
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

import web_studio as W
from core import features as F


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.server = W.ThreadingHTTPServer(("127.0.0.1", 0), W.StudioHandler)
        self.port = self.server.server_port
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def call(self, method, path, body=None):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    def wait_job(self, job_id):
        for _ in range(200):
            status, job = self.call("GET", f"/api/job/{job_id}")
            self.assertEqual(status, 200)
            if job["state"] in ("done", "error"):
                return job
            time.sleep(0.02)
        self.fail("job không kết thúc")


class ModulRiengTest(ServerTest):
    def test_thieu_module_thi_route_tra_404_va_server_van_chay(self):
        vang = {"video": False, "youtube": False, "agy": False, "ffmpeg": True}
        with mock.patch.object(W, "module_presence", return_value=vang):
            for method, path in (("GET", "/api/youtube/status"), ("GET", "/api/agy/status"),
                                 ("GET", "/api/brand"), ("GET", "/api/video/sources"),
                                 ("POST", "/api/video/render"), ("POST", "/api/youtube/upload"),
                                 ("POST", "/api/video/ai")):
                status, body = self.call(method, path, {} if method == "POST" else None)
                self.assertEqual(status, 404, path)
                self.assertIn("error", body)
            status, _ = self.call("GET", "/api/presets")
            self.assertEqual(status, 200)  # tab Âm thanh không bị ảnh hưởng

    def test_route_present_theo_tung_module(self):
        co = {"video": True, "youtube": True, "agy": True, "ffmpeg": True}
        self.assertTrue(W.route_present("/api/video/ai", co))
        self.assertFalse(W.route_present("/api/video/ai", {**co, "agy": False}))  # ai cần cả video lẫn agy
        self.assertFalse(W.route_present("/api/video/sources", {**co, "video": False}))
        self.assertTrue(W.route_present("/api/render", {**co, "video": False, "youtube": False}))
        self.assertTrue(W.route_present("/api/features", {"video": False, "youtube": False,
                                                          "agy": False, "ffmpeg": False}))

    def test_module_presence_thu_muc_trong_thi_tat_het(self):
        with tempfile.TemporaryDirectory() as tmp:
            vang = W.module_presence(root=tmp)
        self.assertFalse(vang["video"])
        self.assertFalse(vang["youtube"])
        self.assertFalse(vang["agy"])


class FeaturesApiTest(ServerTest):
    def test_features_tra_du_truong(self):
        status, body = self.call("GET", "/api/features")
        self.assertEqual(status, 200)
        for key in ("video", "youtube", "agy", "ffmpeg"):
            self.assertIsInstance(body[key], bool)
        for ten in ("stems", "restore"):
            self.assertIn("co", body[ten])
            self.assertIn("dang_cai", body[ten])
            self.assertIn("gpu", body[ten])
        for ten in ("stems", "restore", "ffmpeg"):
            self.assertIn(ten, body["mo_ta"])

    def test_ten_khong_co_thi_404(self):
        status, _ = self.call("POST", "/api/features/khong_co/install", {"bien_the": "cpu"})
        self.assertEqual(status, 404)

    def test_bien_the_sai_thi_400(self):
        status, _ = self.call("POST", "/api/features/stems/install", {"bien_the": "gpu"})
        self.assertEqual(status, 400)

    def test_cuda_khi_khong_co_nvidia_thi_400(self):
        with mock.patch.object(F, "gpu_co", return_value=False):
            status, body = self.call("POST", "/api/features/stems/install", {"bien_the": "cuda"})
        self.assertEqual(status, 400)
        self.assertIn("NVIDIA", body["error"])

    def test_da_co_san_thi_409(self):
        with mock.patch.object(F, "trang_thai", return_value={"co": True, "dang_cai": False, "loi": None, "gpu": False}):
            status, _ = self.call("POST", "/api/features/stems/install", {"bien_the": "cpu"})
        self.assertEqual(status, 409)

    def test_cai_loi_thi_job_bao_loi_va_server_van_chay(self):
        def pip_fail(*args, **kwargs):
            raise RuntimeError("pip lỗi: test")
        with mock.patch.object(F, "trang_thai", return_value={"co": False, "dang_cai": False, "loi": None, "gpu": False}), \
                mock.patch.object(F, "cai", side_effect=pip_fail):
            status, body = self.call("POST", "/api/features/stems/install", {"bien_the": "cpu"})
            self.assertEqual(status, 200)
            job = self.wait_job(body["job_id"])
        self.assertEqual(job["state"], "error")
        self.assertIn("pip lỗi", job["error"])
        status, _ = self.call("GET", "/api/features")  # server vẫn trả lời
        self.assertEqual(status, 200)

    def test_cai_xong_thi_job_done_va_reset_kiem_tra_stem(self):
        def cai_ok(ten, bien_the, tien):
            tien("Cài thư viện đã ghim", 0.5)
        W._stem_support.update(checked=True, available=False)
        with mock.patch.object(F, "trang_thai", return_value={"co": False, "dang_cai": False, "loi": None, "gpu": False}), \
                mock.patch.object(F, "cai", side_effect=cai_ok):
            status, body = self.call("POST", "/api/features/stems/install", {"bien_the": "cpu"})
            self.assertEqual(status, 200)
            job = self.wait_job(body["job_id"])
        self.assertEqual(job["state"], "done")
        self.assertEqual(job["progress"], 1.0)
        self.assertFalse(W._stem_support["checked"])

    def test_ffmpeg_cai_qua_cung_duong_cai(self):
        with mock.patch.object(F, "ffmpeg_co", return_value=False), \
                mock.patch.object(F, "cai_ffmpeg", side_effect=lambda tien: tien("winget", 0.1)):
            status, body = self.call("POST", "/api/features/ffmpeg/install", {})
            self.assertEqual(status, 200)
            job = self.wait_job(body["job_id"])
        self.assertEqual(job["state"], "done")


if __name__ == "__main__":
    unittest.main()
