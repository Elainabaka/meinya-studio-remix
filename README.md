# Studio Remix

Công cụ dòng lệnh (Python) remix âm thanh: biến một bài hát thành bản Nightcore, Slowed + Reverb, Sped Up, Vaporwave, Lo-Fi, Bass Boost hay 8D. Chuỗi xử lý gồm đổi tốc độ (kiểu vinyl hoặc giữ tông), định hình transient, de-esser, EQ và bão hòa, reverb kiểu Abbey Road, rồi mastering về mức LUFS mục tiêu với trần true-peak.

> Repo này không kèm nhạc. Chỉ xử lý bài bạn có quyền dùng.

## Yêu cầu

- Python 3 (đã kiểm trên Python 3.13, Windows).
- FFmpeg trong `PATH`, để đọc và ghi MP3, M4A.

## Cài đặt

```bash
pip install -r requirements.txt
```

## Cách dùng

```bash
# Xem danh sách preset
python remix_cli.py --list

# Một bài, xuất WAV
python remix_cli.py "songs/bai_hat.mp3" --preset slowed_reverb --output "out/bai_hat_slowed.wav"

# Một bài, xuất MP3 320k
python remix_cli.py "songs/bai_hat.mp3" --preset nightcore -f mp3 --bitrate 320k --output "out/bai_hat_nightcore.mp3"

# Cả thư mục
python remix_cli.py "songs/" --preset vaporwave --output "out/"
```

Tuỳ chỉnh thêm: `--speed 0.9`, `--pitch -2` (đặt `--pitch` sẽ chuyển sang chế độ khóa tông, tốc độ và tông tách rời), `--reverb 0.35`, `--bass 3`.

## Preset

| Mã (`--preset`) | Tên | Đặc tính |
|---|---|---|
| `slowed_reverb` | Slowed + Reverb | Chậm 0.85×, reverb sâu, ducking để giữ trống và giọng |
| `nightcore` | Nightcore | Nhanh 1.25×, kick đanh, bass bù lại |
| `sped_up` | Sped Up | Nhanh 1.33×, transient trống rõ |
| `bass_boost` | Drift Phonk & Bass | Bass mạnh, nén sidechain, sub giữ mono |
| `lofi_chill` | Lo-Fi | Chậm 0.92×, dải cao mềm, tape flutter |
| `spatial_8d` | 8D | Âm thanh xoay quanh đầu, bass giữ ở giữa |
| `vaporwave` | Vaporwave | Chậm 0.75×, wow và flutter, reverb trung tâm thương mại |

Thêm preset: tạo file trong `presets/`, kế thừa `RemixPreset` rồi đăng ký trong `presets/registry.py`.

## Tính năng không có trong bản công khai

- `--restore` (bù dải cao bị codec cắt bằng model Apollo): bản công khai không có tính năng này. Lệnh in thông báo và bỏ qua bước đó, không báo lỗi.
- `--stems` (tách stem bằng Demucs) cần thêm `torch` và `demucs`, xem các dòng chú thích trong `requirements.txt`.

## Test

```bash
python -X utf8 -m unittest tests.test_engine -v
```

## Thư viện và giấy phép

Mã của repo: **GPL-3.0**, xem [LICENSE](LICENSE).

Lý do: thư viện `pedalboard` (dùng cho reverb và nén) có giấy phép GPLv3, nên repo dùng GPL-3.0 để khớp với nó.

| Thư viện | Phiên bản | Giấy phép |
|---|---|---|
| numpy | 2.2.6 | BSD |
| scipy | 1.17.1 | BSD |
| soundfile | 0.14.0 | BSD-3-Clause |
| pyloudnorm | 0.2.0 | MIT |
| pedalboard | 0.9.25 | GPLv3 |

---

## English

Command-line audio remix tool in Python. Turns a song into a Nightcore, Slowed + Reverb, Sped Up, Vaporwave, Lo-Fi, Bass Boost or 8D version. Processing: speed change (vinyl style, or key-locked), transient shaping, de-essing, EQ and saturation, Abbey Road style reverb, then mastering to a target LUFS with a true-peak ceiling.

> This repo does not include any music. Only process tracks you have the right to use.

- Requirements: Python 3 (tested on 3.13, Windows), FFmpeg on `PATH`.
- Install: `pip install -r requirements.txt`.
- Run: `python remix_cli.py "song.mp3" --preset nightcore --output "out.wav"`, or `python remix_cli.py --list`.
- Not included in this public version: `--restore` (Apollo model). Using it prints a notice and skips that step.
- Test: `python -X utf8 -m unittest tests.test_engine -v`.
- License: GPL-3.0, because the dependency `pedalboard` is GPLv3. Dependency licenses are listed above.
