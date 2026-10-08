'use strict';

const $ = (id) => document.getElementById(id);
const PREVIEW_SEC = 30;

// Three players: A = original, B = current remix, C = pinned remix (reference for blind tests).
// data = { url, lufs, speed, peaks } ; speed maps slot time <-> original time (t_orig = t * speed).
const slots = {
  A: { audio: $('audioA'), data: null },
  B: { audio: $('audioB'), data: null },
  C: { audio: $('audioC'), data: null },
};
const SLOT_COLOR = { A: '--a-color', B: '--accent', C: '--c-color' };

// Vietnamese labels; presets not listed here fall back to the registry text (plug & play).
const PRESET_VI = {
  slowed_reverb: { name: 'Slowed + Reverb', desc: 'Chậm 0.85×, tông trầm theo, reverb Abbey Road dày và ấm.' },
  nightcore: { name: 'Nightcore', desc: 'Nhanh 1.25×, tông cao theo, trống đanh, sáng, có khử xì.' },
  sped_up: { name: 'Sped Up', desc: 'Nhanh 1.33× kiểu TikTok, trống nảy, khô, có khử xì.' },
  bass_boost: { name: 'Phonk / Bass Boost', desc: 'Giữ tốc độ, sub 65 Hz đậm, nén dày, bass mono chắc.' },
  lofi_chill: { name: 'Lo-Fi Chill', desc: 'Chậm nhẹ 0.92×, ấm như băng cassette, treble mềm.' },
  vaporwave: { name: 'Vaporwave', desc: 'Rất chậm 0.75×, trầm sâu, rung băng từ, vang mênh mông.' },
  spatial_8d: { name: '8D Audio', desc: 'Âm thanh xoay 360° quanh đầu. Cần đeo tai nghe.' },
};

const dB = (v) => `${signed(v)} dB`;
const KNOBS = [
  { key: 'speed', label: 'Tốc độ', min: 0.5, max: 1.6, step: 0.01, fmt: (v) => `${v.toFixed(2)}×`, hint: 'Dưới 1 là chậm, trên 1 là nhanh' },
  { key: 'pitch', label: 'Cao độ', min: -12, max: 12, step: 0.1, fmt: (v) => `${signed(v)} nửa cung`, hint: 'Đơn vị nửa cung. Chỉ chỉnh riêng được khi Khóa tông' },
  { key: 'reverb', label: 'Reverb', min: 0, max: 0.8, step: 0.01, fmt: (v) => `${Math.round(v * 100)}%`, hint: 'Độ vang, đã lọc bass và chói' },
  { key: 'bass_db', label: 'Bass', min: -6, max: 10, step: 0.5, fmt: dB, hint: 'Lực đập dải trầm' },
  { key: 'air_db', label: 'Độ sáng', min: -6, max: 6, step: 0.5, fmt: dB, hint: 'Dải "air" trên 10 kHz. Cao quá dễ chói' },
  { key: 'punch', label: 'Độ đanh trống', min: -0.5, max: 0.8, step: 0.05, fmt: (v) => `${signed(v * 100, 0)}%`, hint: 'Tăng/giảm cú đánh của kick, snare' },
  { key: 'deess_db', label: 'Khử xì', min: 0, max: 12, step: 0.5, fmt: (v) => (v <= 0 ? 'Tắt' : `tối đa −${v.toFixed(1)} dB`), hint: 'Gọt tiếng "s", "x" chói, chỉ khi chúng xuất hiện' },
  { key: 'width', label: 'Độ rộng stereo', min: 0.6, max: 1.8, step: 0.05, fmt: (v) => `${Math.round(v * 100)}%`, hint: 'Bass dưới 110 Hz luôn giữ mono' },
  { key: 'lufs', label: 'Độ to', min: -16, max: -7, step: 0.5, fmt: (v) => `${v.toFixed(1)} LUFS`, hint: 'Spotify/YouTube tự hạ về khoảng -14. To quá chỉ mất độ động' },
];

const MODE_HINT = {
  vinyl: 'Như vặn tốc độ đĩa than: tông đi theo tốc độ. Sạch nhất, đúng chất nightcore/slowed.',
  lock: 'Chỉnh tốc độ và tông độc lập (Rubber Band R3, giữ formant). Có thể hơi "phasey".',
};

const state = {
  presets: [], tracks: [], trackId: null, info: null,
  preset: null, knobs: {}, keyLock: false, pitch: 0, stems: false, restore: false, features: null,
  previewStart: 0, result: null, active: 'B',
  busy: false, pendingPreview: false, exporting: null,   // exporting: promise of the background full export
  pinned: null,                                   // { start, settings }
  blind: { on: false, map: { 1: 'B', 2: 'C' }, tally: { B: 0, C: 0 }, rounds: 0 },
};

// ---------------------------------------------------------------- utils
function signed(v, digits = 1) {
  const s = Math.abs(v).toFixed(digits);
  if (Number(s) === 0) return (0).toFixed(digits);
  return (v > 0 ? '+' : '−') + s;
}
function fmtTime(s) {
  if (!isFinite(s) || s < 0) s = 0;
  return `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
}
const vinylPitch = (speed) => 12 * Math.log2(speed);
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const baseName = (name) => name.replace(/\.[^.]+$/, '');
const presetName = (slug) => (PRESET_VI[slug] || state.presets.find((p) => p.slug === slug) || { name: slug }).name;

async function api(url, opts) {
  const res = await fetch(url, opts);
  const data = await res.json().catch(() => ({}));
  if (!res.ok || data.error) throw new Error(data.error || res.statusText);
  return data;
}
const postJson = (url, body) => api(url, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

function setStatus(text, kind = '') {
  const el = $('statusText');
  el.textContent = text;
  el.className = `status ${kind}`;
}
const setProgress = (frac) => { $('jobBar').style.width = `${(frac * 100).toFixed(1)}%`; };

const STAGE_VI = [
  ['Loading Restore', 'Nạp model phục hồi dải cao'], ['Restoring Highs', 'AI đang bù dải cao bị cắt'],
  ['Loading Stem', 'Nạp model tách stem'], ['Separating Stems', 'AI đang tách giọng / trống / bass'],
  ['Loading', 'Đọc âm thanh'], ['Gain Staging', 'Chuẩn hóa mức vào'], ['Time & Pitch', 'Đổi tốc độ / tông'],
  ['Transient', 'Làm đanh trống'], ['De-esser', 'Khử xì'], ['Sweeten', 'EQ nịnh tai + mono bass'],
  ['Reverb', 'Reverb Abbey Road'], ['8D', 'Xoay 8D'], ['Limiter', 'Master + limiter'],
  ['Mastering', 'Master + limiter'], ['Saving', 'Lưu file'],
];
const stageVi = (s) => (STAGE_VI.find(([k]) => s.includes(k)) || [null, s])[1];

// ---------------------------------------------------------------- presets & knobs
function renderPresets() {
  const list = $('presetList');
  list.innerHTML = '';
  for (const p of state.presets) {
    const btn = document.createElement('button');
    btn.className = `preset${state.preset && p.slug === state.preset.slug ? ' active' : ''}`;
    btn.innerHTML = '<span class="preset-name"></span><span class="preset-desc"></span>';
    btn.querySelector('.preset-name').textContent = presetName(p.slug);
    btn.querySelector('.preset-desc').textContent = PRESET_VI[p.slug] ? PRESET_VI[p.slug].desc : p.description;
    btn.onclick = () => selectPreset(p.slug, true);
    list.appendChild(btn);
  }
}

function selectPreset(slug, rerender) {
  state.preset = state.presets.find((p) => p.slug === slug) || state.presets[0];
  resetKnobs();
  renderPresets();
  if (rerender) maybeAutoPreview();
}

function resetKnobs() {
  const k = state.preset.knobs;
  state.knobs = { ...k };
  state.keyLock = k.pitch !== null;
  state.pitch = state.keyLock ? k.pitch : Math.round(vinylPitch(k.speed) * 10) / 10;
  renderKnobs();
}

function renderKnobs() {
  const wrap = $('knobs');
  wrap.innerHTML = '';
  for (const def of KNOBS) {
    const row = document.createElement('div');
    row.className = 'knob';
    row.dataset.key = def.key;
    row.innerHTML = '<div class="knob-head"><label></label><output></output></div><input type="range"><div class="knob-hint"></div>';
    row.querySelector('label').textContent = def.label;
    row.querySelector('.knob-hint').textContent = def.hint || '';
    const input = row.querySelector('input');
    Object.assign(input, { min: def.min, max: def.max, step: def.step });
    input.setAttribute('aria-label', def.label);
    input.value = def.key === 'pitch' ? state.pitch : state.knobs[def.key];
    input.addEventListener('input', () => {
      const v = parseFloat(input.value);
      if (def.key === 'pitch') state.pitch = v; else state.knobs[def.key] = v;
      updateKnobLabels();
    });
    input.addEventListener('change', maybeAutoPreview);
    wrap.appendChild(row);
  }
  updateKnobLabels();
}

function updateKnobLabels() {
  for (const def of KNOBS) {
    const row = document.querySelector(`.knob[data-key="${def.key}"]`);
    if (!row) continue;
    const out = row.querySelector('output');
    const input = row.querySelector('input');
    if (def.key === 'pitch') {
      const follows = !state.keyLock;
      input.disabled = follows;
      const v = follows ? vinylPitch(state.knobs.speed) : state.pitch;
      if (follows) input.value = v;
      out.textContent = `${def.fmt(v)}${follows ? ' · theo tốc độ' : ''}`;
      const base = state.preset.knobs.pitch;
      row.classList.toggle('changed', state.keyLock && (base === null || Math.abs(state.pitch - base) > 1e-9));
    } else {
      out.textContent = def.fmt(state.knobs[def.key]);
      row.classList.toggle('changed', Math.abs(state.knobs[def.key] - state.preset.knobs[def.key]) > 1e-9);
    }
  }
  $('modeVinyl').classList.toggle('active', !state.keyLock);
  $('modeLock').classList.toggle('active', state.keyLock);
  $('modeHint').textContent = state.keyLock ? MODE_HINT.lock : MODE_HINT.vinyl;
}

function setKeyLock(on) {
  if (state.keyLock === on) return;
  state.keyLock = on;
  if (on) {
    state.pitch = Math.round(vinylPitch(state.knobs.speed) * 10) / 10;
    document.querySelector('.knob[data-key="pitch"] input').value = state.pitch;
  }
  updateKnobLabels();
  maybeAutoPreview();
}

const knobsPayload = () => ({ ...state.knobs, pitch: state.keyLock ? state.pitch : null });

function settingsSummary(st) {
  if (!st) return '';
  const preset = state.presets.find((p) => p.slug === st.preset);
  const parts = [presetName(st.preset)];
  for (const def of KNOBS) {
    const v = st.knobs[def.key];
    const base = preset ? preset.knobs[def.key] : undefined;
    if (def.key === 'pitch') {
      if (v !== base) parts.push(v === null ? 'Vinyl' : `Khóa tông ${def.fmt(v)}`);
    } else if (base === undefined || Math.abs(v - base) > 1e-9) {
      parts.push(`${def.label} ${def.fmt(v)}`);
    }
  }
  if (st.stems) parts.push('Stem AI');
  if (st.restore) parts.push('Bù dải cao AI');
  return parts.join(' · ');
}

// ---------------------------------------------------------------- stem mode
// The server scans every song for a codec lowpass (info.source_cutoff, Hz; null = full band) and the hint says
// whether the AI has anything to restore. A full-band source is skipped by the server even with the switch on.
function updateRestoreUI() {
  const cb = $('restoreMode');
  const hint = $('restoreHint');
  const info = state.info;
  const khz = info && info.source_cutoff ? (info.source_cutoff / 1000).toFixed(1).replace('.', ',') : null;
  $('btnInstallRestore').hidden = !lacks('restore');
  if (lacks('restore')) {
    cb.disabled = true; cb.checked = false; state.restore = false;
    hint.textContent = khz ? `Nguồn bị cắt ở ${khz} kHz, nhưng AI bù chưa cài. Bấm Tải về để bật.`
      : 'AI bù dải cao chưa cài. Bấm Tải về để bật.';
    return;
  }
  if (!info) { cb.disabled = true; hint.textContent = 'Chọn bài hát trước'; return; }
  if (!info.restore_available) {
    cb.disabled = true; cb.checked = false; state.restore = false;
    hint.textContent = khz ? `Nguồn bị cắt ở ${khz} kHz, nhưng chưa có model Apollo (xem README).` : 'Chưa có model Apollo (xem README).';
    return;
  }
  cb.disabled = state.busy;
  cb.checked = state.restore;
  if (!khz) {
    hint.textContent = 'Đã quét: nguồn đủ dải cao, không cần bật (bật cũng tự bỏ qua).';
  } else if (!state.restore) {
    hint.textContent = `Đã quét: nguồn bị cắt ở ${khz} kHz (kiểu MP3 128k). Nên bật để AI bù phần bị mất.`;
  } else if (info.restore_ready) {
    hint.textContent = `AI đang bù dải trên ${khz} kHz, phần còn lại của bài giữ nguyên.`;
  } else {
    hint.textContent = `Nguồn bị cắt ở ${khz} kHz. Lần đầu AI xử lý bài này mất khoảng 1–1,5 phút, sau đó tức thì.`;
  }
}

function updateStemUI() {
  updateRestoreUI();
  const cb = $('stemMode');
  const hint = $('stemHint');
  const info = state.info;
  $('btnInstallStems').hidden = !lacks('stems');
  if (lacks('stems')) {
    cb.disabled = true; cb.checked = false; state.stems = false;
    hint.textContent = 'Stem AI chưa cài. Bấm Tải về để bật (tùy chọn).';
    return;
  }
  if (!info) { cb.disabled = true; hint.textContent = 'Chọn bài hát trước'; return; }
  if (!info.stems_available) {
    cb.disabled = true; cb.checked = false; state.stems = false;
    hint.textContent = 'Chưa cài thư viện AI: pip install demucs';
    return;
  }
  cb.disabled = state.busy;
  cb.checked = state.stems;
  if (!state.stems) {
    hint.textContent = 'AI tách giọng / trống / bass / nhạc cụ để xử lý riêng từng phần.';
  } else if (info.stems_ready) {
    hint.textContent = 'Reverb chỉ phủ giọng + nhạc cụ, trống/bass khô. Làm đanh chỉ trống, khử xì chỉ giọng.';
  } else {
    const where = info.stems_device === 'cuda' ? 'GPU' : 'CPU, sẽ lâu hơn';
    hint.textContent = `Lần đầu AI tách bài này mất khoảng 30–60 giây (${where}), sau đó tức thì.`;
  }
}

// ---------------------------------------------------------------- tracks
function renderTracks() {
  const ul = $('trackList');
  ul.innerHTML = '';
  for (const t of state.tracks) {
    const li = document.createElement('li');
    li.className = `track${t.id === state.trackId ? ' active' : ''}`;
    li.innerHTML = '<span class="track-name"></span><span class="track-src"></span>';
    li.querySelector('.track-name').textContent = baseName(t.name);
    li.querySelector('.track-src').textContent = t.source === 'upload' ? 'đã tải lên' : 'nhạc test';
    li.title = t.name;
    li.onclick = () => selectTrack(t.id);
    ul.appendChild(li);
  }
}

function clearSlots() {
  for (const slot of Object.values(slots)) {
    slot.audio.pause();
    slot.audio.removeAttribute('src');
    slot.data = null;
  }
}

async function selectTrack(id) {
  if (state.busy || id === state.trackId) return;
  clearSlots();
  unpin(false);
  Object.assign(state, { trackId: id, info: null, result: null, active: 'B' });
  renderTracks(); renderMetrics(); updateABUI(); drawAll(); updateButtons(); updateStemUI();
  if (!state.exporting) $('exportResult').innerHTML = '';
  const t = state.tracks.find((x) => x.id === id);
  $('trackTitle').textContent = baseName(t.name);
  $('trackMeta').textContent = 'Đang đọc file...';
  setStatus('Đang giải mã bài hát...', 'busy');
  try {
    const info = await api(`/api/track/${id}`);
    if (state.trackId !== id) return;
    state.info = info;
    state.previewStart = info.suggested_start;
    const meta = $('trackMeta');
    meta.textContent = `${fmtTime(info.duration)} · ${info.sample_rate / 1000} kHz · ${info.lufs.toFixed(1)} LUFS · peak ${signed(info.peak)} dBFS`;
    if (info.peak >= -0.01) {
      const warn = document.createElement('span');
      warn.className = 'warn';
      warn.textContent = ' · bản gốc đã bị clip';
      meta.appendChild(warn);
    }
    setStatus('Sẵn sàng. Bấm "Nghe thử 30 giây" (đã tự chọn đoạn cao trào)');
  } catch (e) {
    $('trackMeta').textContent = 'Không đọc được file';
    setStatus(e.message, 'error');
  }
  drawAll(); updateButtons(); updateStemUI();
}

async function uploadFile(file) {
  if (!file) return;
  setStatus(`Đang tải lên ${file.name}...`, 'busy');
  try {
    const t = await api('/api/upload', {
      method: 'POST', headers: { 'X-Filename': encodeURIComponent(file.name) }, body: file,
    });
    state.tracks = await api('/api/tracks');
    state.trackId = null;
    renderTracks();
    await selectTrack(t.id);
  } catch (e) {
    setStatus(e.message, 'error');
  }
}

// ---------------------------------------------------------------- render jobs
// Preview: plays as soon as the server has the audio (true-peak arrives a moment later).
async function render(mode) {
  if (mode === 'full') return exportFull();
  if (!state.info || !state.preset) return;
  if (state.busy) { state.pendingPreview = true; return; }
  state.busy = true;
  state.pendingPreview = false;
  updateButtons(); updateStemUI();
  setProgress(0.03);
  const settings = { preset: state.preset.slug, knobs: knobsPayload(), stems: state.stems, restore: state.restore };
  const firstSplit = state.stems && !state.info.stems_ready;
  const firstRestore = state.restore && state.info.source_cutoff && !state.info.restore_ready;
  setStatus(firstRestore ? 'AI đang bù dải cao lần đầu (khoảng 1–1,5 phút)...'
    : firstSplit ? 'AI đang tách stem lần đầu (khoảng 30–60 giây)...' : 'Đang render đoạn nghe thử...', 'busy');
  try {
    const { job_id: jobId } = await postJson('/api/render', {
      track_id: state.trackId, preset: settings.preset, knobs: settings.knobs, stems: settings.stems, restore: settings.restore,
      mode: 'preview', start: state.previewStart, format: 'wav',
    });
    let early = null;
    const job = await pollJob(jobId, {
      onAudio: (r) => { early = { ...r, settings }; loadResult(early); },
    });
    if (settings.stems) state.info.stems_ready = true;
    if (job.result.restore) state.info.restore_ready = true;
    if (early && state.result === early) {
      Object.assign(early, job.result);      // same files: only the true-peak is new
      renderMetrics();
    } else if (!early) {
      loadResult({ ...job.result, settings });
    }
    setStatus(`Xong trong ${job.result.elapsed}s. Bấm A/B (phím X) để so`, 'ok');
  } catch (e) {
    setStatus(`Lỗi: ${e.message}`, 'error');
  } finally {
    state.busy = false;
    updateButtons(); updateStemUI();
    setTimeout(() => { if (!state.busy) setProgress(0); }, 700);
    if (state.pendingPreview) render('preview');
  }
}

// Full song: runs in the background (the server keeps previews working meanwhile), the A/B players stay as they are
function exportFull() {
  if (!state.info || !state.preset) return Promise.resolve();
  if (state.exporting) return state.exporting;
  const track = state.trackId;
  const name = baseName(state.tracks.find((x) => x.id === track).name);
  const settings = { preset: state.preset.slug, knobs: knobsPayload(), stems: state.stems, restore: state.restore };
  const el = $('exportResult');
  const show = (text) => { el.textContent = text; };
  show(`Đang xuất cả bài "${name}"...`);
  state.exporting = (async () => {
    try {
      const { job_id: jobId } = await postJson('/api/render', {
        track_id: track, preset: settings.preset, knobs: settings.knobs, stems: settings.stems, restore: settings.restore,
        mode: 'full', start: 0, format: $('fmt').value,
      });
      const job = await pollJob(jobId, {
        onProgress: (j) => show(`Đang xuất cả bài "${name}" · ${stageVi(j.stage || '')} ${Math.round((j.progress || 0) * 100)}%`),
      });
      if (settings.stems && state.trackId === track) state.info.stems_ready = true;
      if (job.result.restore && state.trackId === track) state.info.restore_ready = true;
      showExport({ ...job.result, settings }, track);
    } catch (e) {
      show(`Xuất cả bài lỗi: ${e.message}`);
    } finally {
      state.exporting = null;
      updateButtons();
    }
  })();
  updateButtons();
  return state.exporting;
}

async function pollJob(id, { onAudio = null, onProgress = null } = {}) {
  let heard = false;
  for (;;) {
    await new Promise((r) => setTimeout(r, onProgress ? 400 : 80));
    const job = await api(`/api/job/${id}`);
    if (job.state === 'done') return job;
    if (job.state === 'error' || job.state === 'cancelled') throw new Error(job.error);
    if (job.state === 'audio' && onAudio && !heard) { heard = true; onAudio(job.result); }
    if (onProgress) { onProgress(job); continue; }
    setProgress(job.progress || 0);
    if (job.stage) $('statusText').textContent = `${stageVi(job.stage)}...`;
  }
}

let autoTimer = null;
function maybeAutoPreview() {
  if (!$('autoPreview').checked || !state.result) return;
  clearTimeout(autoTimer);
  autoTimer = setTimeout(() => render('preview'), 150);
}

function loadResult(r) {
  const prev = state.result;
  const cur = activeAudio();
  const wasPlaying = !!cur.src && !cur.paused;
  // Keep the musical position when re-rendering the same section while tuning
  const samePlace = prev && prev.mode === r.mode && prev.start === r.start;
  const origPos = samePlace ? currentOrigTime() : 0;
  if (state.pinned && (r.mode !== 'preview' || r.start !== state.pinned.start)) unpin(false);
  if (state.active === 'C' && !state.pinned) state.active = 'B';
  resetBlindTally();

  state.result = r;
  slots.A.data = { url: r.orig_url, lufs: r.orig_lufs, speed: 1, peaks: r.orig_peaks };
  slots.B.data = { url: r.remix_url, lufs: r.remix_lufs, speed: r.speed, peaks: r.remix_peaks };
  slots.A.audio.src = r.orig_url;
  slots.B.audio.src = r.remix_url;
  applyVolumes();
  const target = activeAudio();
  const t = origPos / slots[state.active].data.speed;
  target.addEventListener('loadedmetadata', () => {
    target.currentTime = Math.min(t, Math.max(0, target.duration - 0.05));
    if (wasPlaying || !prev) target.play().catch(() => {});
  }, { once: true });
  renderMetrics(); updateABUI(); drawAll(); updateButtons();
}

// Tab 2 works on a finished remix: export the current settings as WAV first when this track has none yet
async function toVideoTab() {
  const has = () => state.lastExport && state.lastExport.track === state.trackId
    && JSON.stringify(state.lastExport.settings) === JSON.stringify({
      preset: state.preset.slug, knobs: knobsPayload(), stems: state.stems, restore: state.restore,
    });
  if (!has() && state.exporting) await state.exporting;   // maybe this very song
  if (!has()) {
    $('fmt').value = 'wav';
    await exportFull();
  }
  if (has() && window.meinyaOpenVideo) window.meinyaOpenVideo(state.lastExport.output_path);
}

function showExport(r, track) {
  state.lastExport = { ...r, track };
  const el = $('exportResult');
  el.innerHTML = '';
  el.append('Đã lưu: ');
  const path = document.createElement('span');
  path.textContent = r.output_path;
  const link = document.createElement('a');
  link.href = r.remix_url;
  link.download = r.output_path.split(/[\\/]/).pop();
  link.textContent = 'Tải về';
  el.append(path, link);
  if (track !== state.trackId) return;
  const listen = document.createElement('a');
  listen.href = '#';
  listen.textContent = 'Nghe cả bài';
  listen.onclick = (e) => {
    e.preventDefault();
    if (state.trackId === track && !state.busy) loadResult(r);
  };
  el.append(listen);
}

// ---------------------------------------------------------------- playback, A/B/C, pin, blind
const activeAudio = () => slots[state.active].audio;

function currentOrigTime() {
  const slot = slots[state.active];
  return slot.data ? slot.audio.currentTime * slot.data.speed : 0;
}

function applyVolumes() {
  const present = Object.values(slots).filter((s) => s.data);
  const match = $('loudMatch').checked || state.blind.on;
  const ref = Math.min(...present.map((s) => s.data.lufs));
  for (const slot of Object.values(slots)) {
    slot.audio.volume = slot.data && match ? Math.min(1, Math.pow(10, (ref - slot.data.lufs) / 20)) : 1;
  }
}

function setActive(which) {
  if (which === state.active || !slots[which].data) return;
  const from = activeAudio();
  const playing = !!from.src && !from.paused;
  const tOrig = currentOrigTime();
  from.pause();
  state.active = which;
  const to = activeAudio();
  const t = tOrig / slots[which].data.speed;
  to.currentTime = isFinite(to.duration) ? Math.min(t, Math.max(0, to.duration - 0.05)) : t;
  if (playing) to.play().catch(() => {});
  updateABUI(); updatePlayIcon(); renderMetrics(); drawAll();
}

function toggleX() {
  if (state.blind.on) setActive(state.active === state.blind.map[1] ? state.blind.map[2] : state.blind.map[1]);
  else if (state.pinned) setActive(state.active === 'C' ? 'B' : 'C');
  else setActive(state.active === 'A' ? 'B' : 'A');
}

async function togglePin() {
  if (state.pinned) { unpin(true); return; }
  const r = state.result;
  if (!r || r.mode !== 'preview') return;
  try {
    const { url } = await postJson('/api/pin', { url: r.remix_url });
    slots.C.data = { ...slots.B.data, url };
    slots.C.audio.src = url;
    state.pinned = { start: r.start, settings: r.settings };
    resetBlindTally();
    applyVolumes(); updateABUI();
    setStatus('Đã ghim bản này thành C. Chỉnh tiếp rồi so B (mới) với C, hoặc bật Nghe mù', 'ok');
  } catch (e) {
    setStatus(`Lỗi ghim: ${e.message}`, 'error');
  }
}

function unpin(announce) {
  if (!state.pinned && !slots.C.data) return;
  slots.C.audio.pause();
  slots.C.audio.removeAttribute('src');
  slots.C.data = null;
  state.pinned = null;
  state.blind.on = false;
  $('blindMode').checked = false;
  if (state.active === 'C') setActive('B');
  applyVolumes(); updateABUI(); renderMetrics(); drawAll();
  if (announce) setStatus('Đã bỏ ghim');
}

function resetBlindTally() {
  Object.assign(state.blind, { tally: { B: 0, C: 0 }, rounds: 0 });
  $('blindResult').textContent = '';
}

function shuffleBlind() {
  state.blind.map = Math.random() < 0.5 ? { 1: 'B', 2: 'C' } : { 1: 'C', 2: 'B' };
}

function setBlind(on) {
  state.blind.on = on && !!state.pinned;
  if (state.blind.on) {
    shuffleBlind();
    setActive(state.blind.map[1]);
  }
  applyVolumes(); updateABUI(); renderMetrics(); drawAll(); updateTime();
}

function vote(n) {
  if (!state.blind.on) return;
  const winner = state.blind.map[n];
  state.blind.tally[winner] += 1;
  state.blind.rounds += 1;
  const label = winner === 'C' ? 'C · bản ghim' : 'B · bản mới';
  const { tally, rounds } = state.blind;
  $('blindResult').textContent =
    `Vòng ${rounds}: bạn chọn ${n} = ${label}. Tổng: bản ghim ${tally.C} – bản mới ${tally.B}. Đã xáo lại cho vòng sau.`;
  const keep = state.blind.map[1] === state.active ? 1 : 2;
  shuffleBlind();
  setActive(state.blind.map[keep]);
  updateABUI();
}

function updateABUI() {
  const blind = state.blind.on;
  const pinned = !!state.pinned;
  $('btnA').hidden = blind;
  $('btnC').hidden = !pinned;
  document.querySelector('.ab-switch').classList.toggle('blind', blind);
  const labelB = $('btnB').querySelector('.ab-label');
  const labelC = $('btnC').querySelector('.ab-label');
  $('btnB').querySelector('.ab-key').textContent = blind ? '1' : 'B';
  $('btnC').querySelector('.ab-key').textContent = blind ? '2' : 'C';
  labelB.textContent = blind ? 'Phiên bản 1' : (pinned ? 'Bản mới' : 'Remix');
  labelC.textContent = blind ? 'Phiên bản 2' : 'Bản ghim';
  const shown = { A: 'A', B: blind ? state.blind.map[1] : 'B', C: blind ? state.blind.map[2] : 'C' };
  $('btnA').classList.toggle('active', state.active === shown.A);
  $('btnB').classList.toggle('active', state.active === shown.B);
  $('btnC').classList.toggle('active', state.active === shown.C);

  $('btnPin').textContent = pinned ? 'Bỏ ghim' : 'Ghim bản này';
  $('btnPin').disabled = !pinned && !(state.result && state.result.mode === 'preview');
  $('blindRow').hidden = !pinned;
  $('voteBox').hidden = !blind;
  const info = $('pinInfo');
  info.innerHTML = '';
  if (pinned) {
    const b = document.createElement('b');
    b.textContent = 'C · Bản ghim: ';
    info.append(b, settingsSummary(state.pinned.settings));
    if (!blind && state.result && state.result.settings) {
      info.append(document.createElement('br'), `B · Bản mới: ${settingsSummary(state.result.settings)}`);
    }
  }
}

function togglePlay() {
  const a = activeAudio();
  if (!a.src) return;
  if (a.paused) a.play().catch(() => {}); else a.pause();
}

function updatePlayIcon() {
  // SVG elements ignore the `hidden` property: toggle display instead
  const playing = !activeAudio().paused;
  $('iconPlay').style.display = playing ? 'none' : '';
  $('iconPause').style.display = playing ? '' : 'none';
}

function updateTime() {
  const a = activeAudio();
  $('timeLabel').textContent = state.blind.on
    ? fmtTime(a.currentTime)
    : `${fmtTime(a.currentTime)} / ${fmtTime(a.duration)}`;
}

function updateButtons() {
  const ready = !!state.info && !state.busy;
  $('btnPreview').disabled = !ready;
  $('btnExport').disabled = !state.info || !!state.exporting;
  $('btnExport').textContent = state.exporting ? 'Đang xuất...' : 'Xuất cả bài';
  $('btnToVideo').disabled = !state.info;
  $('btnPlay').disabled = !state.result;
  $('btnPreview').textContent = state.result ? 'Render lại đoạn nghe thử' : 'Nghe thử 30 giây';
  const full = state.result && state.result.mode === 'full';
  $('overviewLabel').textContent = full ? 'Toàn bài · bấm để tua' : 'Toàn bài · bấm để chọn đoạn nghe thử 30 giây';
  if (state.info && !full) {
    const end = Math.min(state.info.duration, state.previewStart + PREVIEW_SEC);
    $('regionLabel').textContent = `Đoạn thử ${fmtTime(state.previewStart)} – ${fmtTime(end)}`;
  } else {
    $('regionLabel').textContent = '';
  }
  updateABUI();
}

function renderMetrics() {
  const el = $('metrics');
  el.innerHTML = '';
  const r = state.result;
  if (!r || state.blind.on) return;
  const items = [
    ['Độ to bản gốc', `${r.orig_lufs.toFixed(1)} LUFS`],
    ['Độ to remix', `${r.remix_lufs.toFixed(1)} LUFS`],
    ['True-peak remix', r.remix_true_peak == null ? 'đang đo...' : `${signed(r.remix_true_peak)} dBTP`],
    ['Tốc độ', `${r.speed.toFixed(2)}×${r.stems ? ' · Stem AI' : ''}${r.restore ? ' · Bù dải cao AI' : ''}`],
    ['Xử lý', `${r.elapsed}s (nhanh ${Math.max(1, Math.round(1 / r.rtf))}× thời gian thực)`],
  ];
  for (const [label, value] of items) {
    const box = document.createElement('div');
    box.className = 'metric';
    box.innerHTML = '<div class="metric-label"></div><div class="metric-value"></div>';
    box.firstChild.textContent = label;
    box.lastChild.textContent = value;
    el.appendChild(box);
  }
  const diff = r.remix_lufs - r.orig_lufs;
  const note = document.createElement('div');
  note.className = 'metric-note';
  note.textContent = $('loudMatch').checked && Math.abs(diff) > 0.1
    ? `Khi so A/B đang hạ ${diff > 0 ? 'bản remix' : 'bản gốc'} ${Math.abs(diff).toFixed(1)} dB để hai bản to bằng nhau, chỉ còn khác nhau về chất âm.`
    : 'Đang nghe đúng độ to thật của từng bản.';
  el.appendChild(note);
}

// ---------------------------------------------------------------- waveforms
function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(c.clientWidth * dpr);
  const h = Math.round(c.clientHeight * dpr);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, w, h);
  return [ctx, w, h, dpr];
}

function drawBars(ctx, peaks, w, h, color) {
  const n = peaks.length;
  const bw = w / n;
  ctx.fillStyle = color;
  for (let i = 0; i < n; i++) {
    const amp = Math.max(0.5, Math.min(1, peaks[i]) * h * 0.46);
    ctx.fillRect(i * bw, h / 2 - amp, Math.max(1, bw * 0.75), amp * 2);
  }
}

function placeholder(ctx, w, h, dpr, text) {
  ctx.fillStyle = cssVar('--muted');
  ctx.font = `${12 * dpr}px Segoe UI, system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(text, w / 2, h / 2);
}

function drawOverview() {
  const [ctx, w, h, dpr] = setupCanvas($('overview'));
  if (!state.info) { placeholder(ctx, w, h, dpr, state.trackId ? 'Đang đọc...' : ''); return; }
  const dur = state.info.duration;
  const r = state.result;
  if (!r || r.mode === 'preview') {
    const x0 = (state.previewStart / dur) * w;
    const x1 = (Math.min(dur, state.previewStart + PREVIEW_SEC) / dur) * w;
    ctx.fillStyle = cssVar('--region');
    ctx.fillRect(x0, 0, x1 - x0, h);
  }
  drawBars(ctx, state.info.peaks, w, h, cssVar('--wave-dim'));
  if (r && activeAudio().src) {
    const pos = (r.mode === 'preview' ? r.start : 0) + currentOrigTime();
    ctx.fillStyle = state.blind.on ? cssVar('--blind') : cssVar(SLOT_COLOR[state.active]);
    ctx.fillRect((pos / dur) * w, 0, 2 * dpr, h);
  }
}

function drawPlayer() {
  const [ctx, w, h, dpr] = setupCanvas($('playerWave'));
  const slot = slots[state.active];
  if (!state.result || !slot.data) {
    placeholder(ctx, w, h, dpr, state.info ? 'Bấm "Nghe thử 30 giây" để nghe bản remix' : '');
    return;
  }
  const a = slot.audio;
  const frac = a.duration ? a.currentTime / a.duration : 0;
  if (state.blind.on) {
    // Waveform shape and length would give the answer away: show only a neutral line
    ctx.fillStyle = cssVar('--wave-dim');
    ctx.fillRect(0, h / 2 - dpr, w, 2 * dpr);
    placeholder(ctx, w, h * 0.55, dpr, 'Nghe mù: chỉ dùng tai. Chọn 1 hay 2 hay hơn');
    return;
  }
  drawBars(ctx, slot.data.peaks, w, h, cssVar('--wave-dim'));
  ctx.save();
  ctx.beginPath();
  ctx.rect(0, 0, frac * w, h);
  ctx.clip();
  drawBars(ctx, slot.data.peaks, w, h, cssVar(SLOT_COLOR[state.active]));
  ctx.restore();
  ctx.fillStyle = cssVar('--text');
  ctx.fillRect(frac * w, 0, 1.5 * dpr, h);
}

function drawAll() { drawOverview(); drawPlayer(); updateTime(); }

function frame() {
  if (!activeAudio().paused) drawAll();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------- tính năng tải khi cần (MD91)
// /api/features: module riêng (video, youtube, agy) có mặt không; stem, restore, ffmpeg đã cài chưa.
// Thiếu module riêng thì ẩn tab và nút tương ứng; thiếu tính năng tải khi cần thì hỏi trước khi tải.
let featOn = null;   // tính năng đang hỏi trong hộp thoại

function lacks(ten) {
  const f = state.features;
  if (!f) return false;   // chưa đọc được danh sách: giữ giao diện gốc
  return typeof f[ten] === 'object' ? !f[ten].co : !f[ten];
}

async function loadFeatures() {
  try { state.features = await api('/api/features'); } catch (e) { state.features = null; return; }
  applyFeatures();
}

function applyFeatures() {
  const f = state.features;
  $('agyChip').hidden = !f.agy;
  for (const b of document.querySelectorAll('[data-tab="video"]')) b.hidden = !f.video;
  $('btnToVideo').hidden = !f.video;
  $('ffmpegBox').hidden = !!f.ffmpeg;
  $('fmt').querySelector('option[value="mp3"]').disabled = !f.ffmpeg;
  if (!f.ffmpeg && $('fmt').value === 'mp3') $('fmt').value = 'wav';
  updateStemUI();
}

function openFeat(ten) {
  const mo = state.features.mo_ta[ten];
  featOn = ten;
  $('featTitle').textContent = mo.ten;
  $('featText').textContent = `Tính năng này cần tải thêm: ${mo.dung_luong} (nguồn ${mo.nguon}; giấy phép ${mo.giay_phep}). Tải về dùng không?`;
  $('featMo').textContent = mo.mo_ta;
  $('featBienRow').hidden = ten === 'ffmpeg' || !state.features.gpu;
  $('featBien').value = state.features.gpu ? 'cuda' : 'cpu';
  $('featBarWrap').hidden = true; $('featStage').hidden = true; $('featErr').hidden = true;
  $('featGo').disabled = false; $('featLater').disabled = false;
  $('featModal').hidden = false;
  $('featGo').focus();
}

function closeFeat() {
  $('featModal').hidden = true;
  featOn = null;
}

async function startFeat() {
  const ten = featOn;
  if (!ten) return;
  $('featGo').disabled = true; $('featLater').disabled = true;
  $('featErr').hidden = true;
  $('featBarWrap').hidden = false; $('featStage').hidden = false;
  $('featBar').style.width = '0%'; $('featStage').textContent = 'Đang gửi yêu cầu...';
  try {
    const { job_id } = await postJson(`/api/features/${ten}/install`, { bien_the: $('featBien').value });
    await pollJob(job_id, {
      onProgress: (job) => {
        $('featBar').style.width = `${((job.progress || 0) * 100).toFixed(1)}%`;
        if (job.stage) $('featStage').textContent = `${job.stage}...`;
      },
    });
  } catch (e) {
    $('featErr').textContent = `Chưa cài được (${e.message}). Tính năng vẫn tắt, app vẫn dùng bình thường.`;
    $('featErr').hidden = false;
    $('featGo').disabled = false; $('featLater').disabled = false;
    return;
  }
  closeFeat();
  if (ten === 'stems') state.stems = true;
  if (ten === 'restore') state.restore = true;
  await loadFeatures();
  await refreshInfo();
  maybeAutoPreview();
  setStatus('Đã cài xong, tính năng đã bật.');
}

// After an install the track must be read again: the server now sees the new libraries
async function refreshInfo() {
  const id = state.trackId;
  if (!id) return;
  try {
    const info = await api(`/api/track/${id}`);
    if (state.trackId === id) state.info = info;
  } catch (e) { /* giữ thông tin cũ */ }
  updateStemUI();
}

// ---------------------------------------------------------------- events
function onSlotButton(btn) {
  if (state.blind.on) setActive(state.blind.map[btn === 'B' ? 1 : 2]);
  else setActive(btn);
}

function bindEvents() {
  $('btnPreview').onclick = () => render('preview');
  $('btnExport').onclick = () => render('full');
  $('btnToVideo').onclick = toVideoTab;
  $('btnPlay').onclick = togglePlay;
  $('btnA').onclick = () => onSlotButton('A');
  $('btnB').onclick = () => onSlotButton('B');
  $('btnC').onclick = () => onSlotButton('C');
  $('btnPin').onclick = togglePin;
  $('blindMode').onchange = (e) => setBlind(e.target.checked);
  $('vote1').onclick = () => vote(1);
  $('vote2').onclick = () => vote(2);
  $('btnReset').onclick = () => { resetKnobs(); maybeAutoPreview(); };
  $('modeVinyl').onclick = () => setKeyLock(false);
  $('modeLock').onclick = () => setKeyLock(true);
  $('loudMatch').onchange = () => { applyVolumes(); renderMetrics(); };
  $('stemMode').onchange = (e) => { state.stems = e.target.checked; updateStemUI(); maybeAutoPreview(); };
  $('restoreMode').onchange = (e) => { state.restore = e.target.checked; updateStemUI(); maybeAutoPreview(); };
  $('btnInstallStems').onclick = () => openFeat('stems');
  $('btnInstallRestore').onclick = () => openFeat('restore');
  $('btnInstallFfmpeg').onclick = () => openFeat('ffmpeg');
  $('featGo').onclick = startFeat;
  $('featLater').onclick = closeFeat;

  for (const slot of Object.values(slots)) {
    for (const ev of ['play', 'pause', 'ended']) slot.audio.addEventListener(ev, () => { updatePlayIcon(); drawAll(); });
    slot.audio.addEventListener('loadedmetadata', drawAll);
    slot.audio.addEventListener('seeked', drawAll);
  }

  $('playerWave').addEventListener('click', (e) => {
    const a = activeAudio();
    if (!a.src || !isFinite(a.duration)) return;
    const rect = e.currentTarget.getBoundingClientRect();
    a.currentTime = ((e.clientX - rect.left) / rect.width) * a.duration;
    drawAll();
  });

  $('overview').addEventListener('click', (e) => {
    if (!state.info) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const t = ((e.clientX - rect.left) / rect.width) * state.info.duration;
    const r = state.result;
    if (r && r.mode === 'full') {
      activeAudio().currentTime = t / slots[state.active].data.speed;
    } else {
      state.previewStart = Math.max(0, Math.min(Math.floor(t), state.info.duration - PREVIEW_SEC));
      updateButtons();
      maybeAutoPreview();
    }
    drawAll();
  });

  const dz = $('dropzone');
  $('fileInput').onchange = (e) => uploadFile(e.target.files[0]);
  dz.addEventListener('dragover', (e) => { e.preventDefault(); dz.classList.add('drag'); });
  dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    dz.classList.remove('drag');
    uploadFile(e.dataTransfer.files[0]);
  });
  // Dropping anywhere else must not navigate away from the studio
  window.addEventListener('dragover', (e) => e.preventDefault());
  window.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
  });

  document.addEventListener('keydown', (e) => {
    if (document.body.dataset.tab !== 'audio' || ['SELECT', 'INPUT', 'TEXTAREA'].includes(e.target.tagName) && e.target.type !== 'range'
      || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
    else if (e.code === 'KeyX') toggleX();
    else if (state.blind.on && (e.code === 'Digit1' || e.code === 'Digit2')) setActive(state.blind.map[e.code === 'Digit1' ? 1 : 2]);
    else if (!state.blind.on && e.code === 'KeyA') setActive('A');
    else if (!state.blind.on && e.code === 'KeyB') setActive('B');
    else if (!state.blind.on && e.code === 'KeyC') setActive('C');
  });

  window.addEventListener('resize', drawAll);
}

async function init() {
  bindEvents();
  const featOk = loadFeatures();
  try {
    [state.presets, state.tracks] = await Promise.all([api('/api/presets'), api('/api/tracks')]);
  } catch (e) {
    setStatus(`Không kết nối được server: ${e.message}`, 'error');
    return;
  }
  await featOk;
  selectPreset(state.presets.some((p) => p.slug === 'slowed_reverb') ? 'slowed_reverb' : state.presets[0].slug, false);
  renderTracks();
  setStatus('Chọn bài hát để bắt đầu');
  drawAll(); updateABUI(); updateStemUI();
  requestAnimationFrame(frame);
  if (state.tracks.length) selectTrack(state.tracks[0].id);
}

init();
