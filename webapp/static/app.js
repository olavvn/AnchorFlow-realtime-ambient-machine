/* ThemeTransformer web client
 * - SSE(/api/stream)로 노트 이벤트 수신 → 피아노롤 시각화
 * - 오디오 없음: loopMIDI 출력은 서버에서 직접 처리
 * - MELODY / PAD 트랙 색상 구분
 */
"use strict";

const $ = (id) => document.getElementById(id);

// ── 피아노롤 파라미터 ─────────────────────────────────────────
const WINDOW_PAST   = 4.0;   // 플레이헤드 뒤로 보일 구간(초)
const WINDOW_FUTURE = 30.0;  // 앞으로 보일 구간(초)
const PRUNE_AGE     = 10.0;  // 이보다 오래된 노트 폐기

// ── 색상 ─────────────────────────────────────────────────────
const COLORS = {
  seed_MELODY:   "#5b8cff",
  seed_PAD:      "#39d98a",
  gen_MELODY:    "#5b8cff",
  gen_PAD:       "#39d98a",
  anchor_MELODY: "#ffb454",
  anchor_PAD:    "#ff7eb6",
};
function noteColor(src, track) {
  return COLORS[src + "_" + track] || "#5b8cff";
}

// ── 상태 ─────────────────────────────────────────────────────
let es       = null;
let running  = false;
let wallStart = null;      // performance.now() 기준 시작 시각(ms)
let noteCount = 0;
const notes   = [];        // {t, pitch, dur, vel, track, src}
const markers = [];        // {t, src, label}

// ── 라이브러리 로드 ──────────────────────────────────────────
async function loadLibrary() {
  const r   = await fetch("/api/library");
  const lib = await r.json();

  $("device-badge").textContent = "device: " + lib.device;

  // MIDI 포트 상태
  const ports = lib.ports || {};
  $("port-melody").textContent = ports.melody || "–";
  $("port-pad").textContent    = ports.pad    || "–";
  if (ports.error) {
    $("midi-badge").textContent = "MIDI: 미연결";
    $("midi-badge").classList.add("err");
  } else {
    $("midi-badge").textContent = "MIDI: 연결됨";
    $("midi-badge").classList.add("ok");
  }

  // Seed 목록
  const ss = $("seed-select");
  ss.innerHTML = "";
  (lib.seeds || []).forEach(s => {
    const o = document.createElement("option");
    o.value = s.id; o.textContent = "▣ " + s.name; ss.appendChild(o);
  });

  // Anchor 칩
  const al = $("anchor-list");
  al.innerHTML = "";
  (lib.anchors || []).forEach(a => {
    al.appendChild(makeChip(a.name, a.id));
  });
}

function makeChip(label, id) {
  const b = document.createElement("button");
  b.className = "chip"; b.textContent = label; b.disabled = true;
  b.addEventListener("click", () => injectAnchor(b, id));
  return b;
}
function setChipsEnabled(on) {
  document.querySelectorAll(".chip").forEach(c => c.disabled = !on);
}

// ── 세션 시작/정지 ───────────────────────────────────────────
async function start() {
  resetState();
  setState("buffering");
  $("hint").textContent = "생성 시작 중…";

  const body = {
    seed_id:     $("seed-select").value,
    temperature: parseFloat($("temp").value),
    top_p:       parseFloat($("topp").value),
    time_scale:  parseFloat($("tscale").value),
    pitch_min:   parseInt($("pmin").value),
    pitch_max:   parseInt($("pmax").value),
    chunk_size:  32,
  };
  const r   = await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const res = await r.json();
  if (!r.ok) {
    setState("error");
    $("hint").textContent = "시작 실패: " + (res.error || "unknown");
    return;
  }

  running   = true;
  // Use server's epoch timestamp so piano roll clock matches MIDI scheduler clock
  wallStart = res.wall_start_epoch * 1000;  // epoch ms
  $("start-btn").disabled   = true;
  $("stop-btn").disabled    = false;
  $("seed-select").disabled = true;
  setChipsEnabled(true);
  setState("playing");
  $("hint").textContent = "재생 중 · Anchor를 누르면 컨텍스트에 리터럴 삽입됩니다.";
  openStream();
}

async function stop() {
  running = false;
  if (es) { es.close(); es = null; }
  try { await fetch("/api/stop", { method: "POST" }); } catch (e) {}
  setState("idle");
  $("start-btn").disabled   = false;
  $("stop-btn").disabled    = true;
  $("seed-select").disabled = false;
  setChipsEnabled(false);
  $("hint").textContent = "정지됨. ▶ 를 눌러 다시 시작하세요.";
}

function resetState() {
  if (es) { es.close(); es = null; }
  notes.length = 0; markers.length = 0;
  wallStart = null; noteCount = 0;
  $("note-count").textContent = "0";
  $("clock").textContent = "0:00";
  $("buf-bar").style.width = "0%";
  $("buf-val").textContent = "0.0s";
}

// ── SSE ──────────────────────────────────────────────────────
function openStream() {
  es = new EventSource("/api/stream");
  es.onmessage = (ev) => {
    let arr; try { arr = JSON.parse(ev.data); } catch (e) { return; }
    if (!Array.isArray(arr)) return;
    arr.forEach(handleEvent);
  };
}

function handleEvent(e) {
  if (e.kind === "note") {
    notes.push(e);
    noteCount++;
    $("note-count").textContent = noteCount;
  } else if (e.kind === "marker") {
    markers.push(e);
  }
}

// ── Anchor 주입 ──────────────────────────────────────────────
async function injectAnchor(btn, id) {
  btn.classList.add("flash");
  setTimeout(() => btn.classList.remove("flash"), 450);
  try {
    await fetch("/api/anchor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ anchor_id: id }),
    });
  } catch (e) {}
}

// ── 상태 뱃지 ────────────────────────────────────────────────
function setState(s) {
  const b = $("state-badge");
  b.className = "badge state-" + s;
  b.textContent = { idle: "idle", buffering: "buffering", playing: "playing", error: "error" }[s] || s;
}

// ── 현재 음악시간 (wall-clock 기준 1:1) ─────────────────────
function musicalNow() {
  if (!wallStart) return 0;
  return (Date.now() - wallStart) / 1000;
}

// ── 피아노롤 렌더링 ──────────────────────────────────────────
const canvas = $("roll"), ctx = canvas.getContext("2d");
let _pruneCount = 0;

function fitCanvas() {
  const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  canvas.width  = r.width  * dpr;
  canvas.height = r.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", fitCanvas);

const PITCH_MIN = 12, PITCH_MAX = 108;

function draw() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);

  const ph   = musicalNow();
  const t0   = ph - WINDOW_PAST, t1 = ph + WINDOW_FUTURE, span = t1 - t0;
  const xOf  = (t) => (t - t0) / span * w;
  const yOf  = (p) => {
    const c = Math.max(PITCH_MIN, Math.min(PITCH_MAX, p));
    return h - (c - PITCH_MIN) / (PITCH_MAX - PITCH_MIN) * h;
  };

  // 옥타브 그리드
  ctx.strokeStyle = "rgba(255,255,255,.04)"; ctx.lineWidth = 1;
  for (let p = PITCH_MIN; p <= PITCH_MAX; p += 12) {
    const y = yOf(p);
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  // 마커
  for (const m of markers) {
    if (m.t < t0 - 1 || m.t > t1) continue;
    const x = xOf(m.t);
    ctx.strokeStyle = m.src === "seed" ? "rgba(91,140,255,.6)" : "rgba(255,180,84,.8)";
    ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "11px Segoe UI";
    ctx.save(); ctx.translate(x + 3, 14); ctx.fillText(m.label || m.src, 0, 0); ctx.restore();
  }

  // 노트
  const noteH = Math.max(3, (h / (PITCH_MAX - PITCH_MIN)) * 1.6);
  for (const n of notes) {
    if (n.t + n.dur < t0 || n.t > t1) continue;
    const x    = xOf(n.t), x2 = xOf(n.t + n.dur), y = yOf(n.pitch);
    const wbar = Math.max(3, x2 - x);
    const col  = noteColor(n.src, n.track);
    const played = n.t <= ph;
    ctx.globalAlpha = played ? 1.0 : 0.5;
    ctx.fillStyle   = col;
    roundRect(ctx, x, y - noteH / 2, wbar, noteH, Math.min(3, noteH / 2));
    ctx.fill();
    if (n.src === "anchor") {
      ctx.globalAlpha = 0.85;
      ctx.shadowColor = col; ctx.shadowBlur = 8;
      roundRect(ctx, x, y - noteH / 2, wbar, noteH, Math.min(3, noteH / 2));
      ctx.fill(); ctx.shadowBlur = 0;
    }
  }
  ctx.globalAlpha = 1;

  // 플레이헤드
  const px = xOf(ph);
  ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();

  // HUD 업데이트
  if (running) {
    const maxT = notes.length ? Math.max(...notes.map(n => n.t + n.dur)) : 0;
    const buf  = Math.max(0, maxT - ph);
    $("buf-val").textContent = buf.toFixed(1) + "s";
    $("buf-bar").style.width = Math.min(100, buf / 20 * 100) + "%";
    $("clock").textContent   = fmt(Math.max(0, ph));
  }

  // 오래된 노트 정리
  if (++_pruneCount % 60 === 0) {
    const cut = ph - PRUNE_AGE;
    while (notes.length   && notes[0].t   + notes[0].dur   < cut) notes.shift();
    while (markers.length && markers[0].t + 1               < cut) markers.shift();
  }

  requestAnimationFrame(draw);
}

function roundRect(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y,     x + w, y + h, r);
  c.arcTo(x + w, y + h, x,     y + h, r);
  c.arcTo(x,     y + h, x,     y,     r);
  c.arcTo(x,     y,     x + w, y,     r);
  c.closePath();
}
function fmt(s) {
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return m + ":" + String(ss).padStart(2, "0");
}

// ── 이벤트 바인딩 ────────────────────────────────────────────
$("start-btn").addEventListener("click", start);
$("stop-btn").addEventListener("click", stop);
$("panic-btn").addEventListener("click", async () => {
  try { await fetch("/api/panic", { method: "POST" }); } catch (e) {}
});
$("temp").addEventListener("input", e => $("temp-val").textContent = parseFloat(e.target.value).toFixed(2));
$("topp").addEventListener("input", e => $("topp-val").textContent = parseFloat(e.target.value).toFixed(2));
$("tscale").addEventListener("input", e => $("tscale-val").textContent = parseFloat(e.target.value).toFixed(2));
$("pmin").addEventListener("input", e => $("pmin-val").textContent = e.target.value);
$("pmax").addEventListener("input", e => $("pmax-val").textContent = e.target.value);

loadLibrary();
fitCanvas();
requestAnimationFrame(draw);
