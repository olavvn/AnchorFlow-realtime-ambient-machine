"""ThemeTransformer 실시간 스트리밍 웹서버.

Seed MIDI → 인코더 theme 조건 → 무한 자기회귀 생성.
Anchor 주입 시 인코더 theme 교체 → 이후 생성이 새 theme 조건으로 전환.
생성된 토큰 → SSE(피아노롤 시각화) + rtmidi loopMIDI(MELODY/PAD 각각 별도 포트).

실행:
    myenv/Scripts/python.exe webapp/server.py \\
        --model-path trained_model/model_ep325.pt \\
        --melody-port "melody" --pad-port "pad"
"""
from __future__ import annotations

import heapq
import itertools
import json
import os
import atexit
import queue
import signal
import sys
import threading
import time
import argparse
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from flask import Flask, Response, jsonify, request, send_from_directory

from preprocess.vocab import Vocab
from mymodel import myLM
from src.streaming.buffer import AnchorQueue, TokenChunkQueue
from src.streaming.chunk_gen import ChunkGenerator

WEBAPP_DIR  = Path(__file__).resolve().parent
STATIC_DIR  = WEBAPP_DIR / "static"
SEEDS_DIR   = WEBAPP_DIR / "assets" / "seeds"
ANCHORS_DIR = WEBAPP_DIR / "assets" / "anchors"


# ─────────────────────────────────────────────────────────────────────────────
# TSD 스트리밍 디코더 (토큰 → 절대 음악시간 노트 이벤트)
# ─────────────────────────────────────────────────────────────────────────────
class TSDStreamingDecoder:
    """청크 경계를 넘어 상태를 유지하는 TSD→노트이벤트 변환기."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        self.t = 0.0
        self._pending: Optional[dict] = None
        self._buf: List[int] = []   # 미완성 Note-On/Duration 보관

    def feed(self, token_ids: List[int], src: str) -> List[dict]:
        toks = self._buf + list(token_ids)
        self._buf = []
        id2 = self.vocab.id2token
        events: List[dict] = []
        n = len(toks)
        i = 0

        while i < n:
            tok = id2.get(toks[i], "padding")

            if tok.startswith("Time-Shift_"):
                steps = int(tok.split("_")[1])
                self.t += steps * self.vocab.time_resolution
                i += 1

            elif tok.startswith("Note-On-"):
                # 뒤따르는 Duration/Velocity가 이 청크에 없을 수 있음 → 보류
                if i + 2 >= n:
                    self._buf = toks[i:]
                    break
                parts = tok.split("_")
                track = parts[0].split("-")[2]   # MELODY or PAD
                pitch = int(parts[1])
                self._pending = {"track": track, "pitch": pitch, "t": round(self.t, 4)}
                i += 1

            elif tok.startswith("Note-Duration-") and self._pending is not None:
                steps = int(tok.split("_")[1])
                self._pending["dur"] = round(steps * self.vocab.time_resolution, 4)
                i += 1

            elif tok.startswith("Note-Velocity-") and self._pending is not None and "dur" in self._pending:
                vel = int(tok.split("_")[1])
                events.append({
                    "kind":  "note",
                    "t":     self._pending["t"],
                    "pitch": self._pending["pitch"],
                    "dur":   self._pending["dur"],
                    "vel":   vel,
                    "track": self._pending["track"],
                    "src":   src,
                })
                self._pending = None
                i += 1

            else:
                i += 1

        if len(self._buf) > 6:
            self._buf = self._buf[-3:]
        return events


# ─────────────────────────────────────────────────────────────────────────────
# 실시간 MIDI 스케줄러 (rtmidi → loopMIDI)
# ─────────────────────────────────────────────────────────────────────────────
class MIDIScheduler:
    """노트 이벤트를 받아 wall-clock 기준으로 rtmidi에 정확히 전송."""

    NOTE_ON  = 0x90
    NOTE_OFF = 0x80

    def __init__(self, melody_port: str, pad_port: str):
        self._melody_name = melody_port
        self._pad_name    = pad_port
        self._melody_out  = None
        self._pad_out     = None
        self._pq: List    = []
        self._cnt         = itertools.count()
        self._lock        = threading.Lock()
        self._stop        = threading.Event()
        self._wall_start  = 0.0
        self._thread: Optional[threading.Thread] = None
        self._port_error  = ""

    def open_ports(self) -> bool:
        try:
            import rtmidi
        except ImportError:
            self._port_error = "python-rtmidi not installed"
            return False
        try:
            self._melody_out = self._open_port(rtmidi.MidiOut(), self._melody_name, "MELODY")
            self._pad_out    = self._open_port(rtmidi.MidiOut(), self._pad_name,    "PAD")
            return True
        except RuntimeError as e:
            self._port_error = str(e)
            return False

    @staticmethod
    def _open_port(out, name_substr: str, label: str):
        ports = out.get_ports()
        for i, p in enumerate(ports):
            if name_substr.lower() in p.lower():
                out.open_port(i)
                print(f"[midi] {label} → [{i}] {p}", flush=True)
                return out
        raise RuntimeError(
            f"MIDI port '{name_substr}' not found. Available: {ports}"
        )

    @staticmethod
    def list_ports() -> List[str]:
        try:
            import rtmidi
            o = rtmidi.MidiOut()
            return o.get_ports()
        except Exception:
            return []

    def start(self, wall_start: float):
        self._wall_start = wall_start
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def panic(self):
        """All Sound Off + All Notes Off on all 16 channels for both ports."""
        for port in (self._melody_out, self._pad_out):
            if port is None:
                continue
            try:
                for ch in range(16):
                    port.send_message([0xB0 | ch, 120, 0])  # All Sound Off
                    port.send_message([0xB0 | ch, 123, 0])  # All Notes Off
            except Exception:
                pass

    def stop(self):
        self._stop.set()
        self.panic()

    def schedule(self, event: dict):
        if event.get("kind") != "note":
            return
        on_at  = self._wall_start + event["t"]
        off_at = on_at + event["dur"]
        c = next(self._cnt)
        with self._lock:
            heapq.heappush(self._pq, (on_at,  c,     "on",  event))
            heapq.heappush(self._pq, (off_at, c+0.5, "off", event))

    def _loop(self):
        while not self._stop.is_set():
            with self._lock:
                if not self._pq:
                    fire_at = None
                else:
                    fire_at = self._pq[0][0]

            if fire_at is None:
                time.sleep(0.002)
                continue

            now = time.perf_counter()
            if fire_at > now:
                time.sleep(min(fire_at - now, 0.005))
                continue

            with self._lock:
                if not self._pq or self._pq[0][0] > time.perf_counter():
                    continue
                fire_at, _, kind, ev = heapq.heappop(self._pq)

            port = self._melody_out if ev["track"] == "MELODY" else self._pad_out
            if port is None:
                continue
            try:
                if kind == "on":
                    port.send_message([self.NOTE_ON,  ev["pitch"], ev["vel"]])
                else:
                    port.send_message([self.NOTE_OFF, ev["pitch"], 0])
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# 생성 세션
# ─────────────────────────────────────────────────────────────────────────────
class WebSession:
    """생성 세션.

    seed_theme_seq  → ChunkGenerator 인코더 입력 (theme).
    anchor 주입 시  → 인코더 theme 교체 (디코더 컨텍스트는 유지).
    디코더는 decoder context만 누적하며 자유 생성.
    """

    def __init__(
        self,
        model,
        vocab: Vocab,
        device,
        seed_tokens: List[int],      # seed MIDI raw tokens (리터럴 재생 + 인코더 theme)
        scheduler: MIDIScheduler,
        *,
        temp: float = 1.2,
        top_p: float = 0.9,
        chunk_size: int = 32,
        max_len: int = 512,
        pitch_min: int = 0,
        pitch_max: int = 127,
        lead_cap: float = 20.0,
    ):
        self.vocab     = vocab
        self.scheduler = scheduler
        self.lead_cap  = lead_cap
        self.sse_q: "queue.Queue[dict]" = queue.Queue()
        self._running  = False

        self._anchor_q = AnchorQueue()
        self._token_q  = TokenChunkQueue(maxsize=8)

        self.decoder = TSDStreamingDecoder(vocab)

        # seed → 인코더 theme (Theme_Start + tokens + Theme_End)
        seed_theme_seq = (
            [vocab.token2id["Theme_Start"]]
            + seed_tokens
            + [vocab.token2id["Theme_End"]]
        )

        self._gen = ChunkGenerator(
            model=model, vocab=vocab, theme_seq=seed_theme_seq, device=device,
            anchor_queue=self._anchor_q, token_queue=self._token_q,
            chunk_size=chunk_size, max_len=max_len,
            temp=temp, top_p=top_p,
            pitch_min=pitch_min, pitch_max=pitch_max,
        )
        # seed → 디코더 컨텍스트 리터럴 삽입
        self._gen.init_seed(seed_tokens)

        self._wall_start: float = 0.0

        # seed 노트를 SSE에 선 방출 (피아노롤 시각화)
        self.sse_q.put({"kind": "marker", "t": 0.0, "src": "seed", "label": "SEED"})
        if seed_tokens:
            for ev in self.decoder.feed(seed_tokens, "seed"):
                self.sse_q.put(ev)

    def start(self):
        self._running = True
        self._wall_start = time.perf_counter()
        self.scheduler.start(self._wall_start)

        # seed 노트를 MIDI 스케줄러에도 등록 (리터럴 재생)
        for item in list(self.sse_q.queue):
            self.scheduler.schedule(item)

        self._gen.start()
        threading.Thread(target=self._reader_loop, daemon=True).start()

    def stop(self):
        self._running = False
        self._gen.stop()
        self.scheduler.stop()

    def feed_anchor(self, label: str, midi_path: str):
        """앵커를 큐에 넣고 SSE 마커 방출. 실제 theme 교체는 generator 스레드에서 처리."""
        self._anchor_q.put(midi_path)
        # 마커 t는 decoder의 현재 음악시간 기준
        self.sse_q.put({"kind": "marker", "t": round(self.decoder.t, 4),
                        "src": "anchor", "label": label})

    def drain_sse(self, timeout: float = 0.25) -> List[dict]:
        out: List[dict] = []
        try:
            out.append(self.sse_q.get(timeout=timeout))
        except queue.Empty:
            return out
        while True:
            try:
                out.append(self.sse_q.get_nowait())
            except queue.Empty:
                break
        return out

    def _reader_loop(self):
        while self._running:
            elapsed = time.perf_counter() - self._wall_start
            if (self.decoder.t - elapsed) > self.lead_cap:
                time.sleep(0.05)
                continue

            try:
                kind, tokens = self._token_q.get(timeout=0.3)
            except queue.Empty:
                continue

            src = "anchor" if kind == "anchor" else "gen"
            events = self.decoder.feed(tokens, src)
            for ev in events:
                self.sse_q.put(ev)
                self.scheduler.schedule(ev)


# ─────────────────────────────────────────────────────────────────────────────
# 모델 홀더
# ─────────────────────────────────────────────────────────────────────────────
class ModelHolder:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vocab  = Vocab()
        print(f"[model] loading {model_path} on {self.device} …", flush=True)
        t0 = time.time()
        self.model = myLM(
            self.vocab.n_tokens, d_model=256,
            num_encoder_layers=6, xorpattern=[0, 0, 0, 1, 1, 1],
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"[model] ready ({time.time()-t0:.1f}s)  vocab={self.vocab.n_tokens}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Flask 앱
# ─────────────────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=None)
HOLDER: ModelHolder
SCHEDULER: MIDIScheduler
SESSION: Optional[WebSession] = None
SESSION_LOCK = threading.Lock()


def _list_midis(d: Path) -> List[dict]:
    if not d.exists():
        return []
    return [{"id": p.name, "name": p.stem} for p in sorted(d.glob("*.mid"))]


def _resolve(category: str, mid_id: str) -> Path:
    base = {"seed": SEEDS_DIR, "anchor": ANCHORS_DIR}[category]
    p = (base / mid_id).resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ValueError("invalid path")
    if not p.exists():
        raise FileNotFoundError(mid_id)
    return p


def _midi_to_tokens(path: Path, vocab: Vocab) -> List[int]:
    return vocab.midi2TSD(str(path), theme_annotations=False)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:fname>")
def static_files(fname):
    return send_from_directory(STATIC_DIR, fname)


@app.route("/api/library")
def library():
    return jsonify({
        "seeds":   _list_midis(SEEDS_DIR),
        "anchors": _list_midis(ANCHORS_DIR),
        "device":  HOLDER.device,
        "ports": {
            "melody": SCHEDULER._melody_name,
            "pad":    SCHEDULER._pad_name,
            "available": MIDIScheduler.list_ports(),
            "error": SCHEDULER._port_error,
        },
    })


@app.route("/api/start", methods=["POST"])
def start():
    global SESSION
    data = request.get_json(force=True)

    seed_id = data.get("seed_id")
    if not seed_id:
        return jsonify({"error": "seed_id is required"}), 400

    try:
        seed_path   = _resolve("seed", seed_id)
        seed_tokens = _midi_to_tokens(seed_path, HOLDER.vocab)
    except Exception as e:
        return jsonify({"error": f"seed load failed: {e}"}), 400

    with SESSION_LOCK:
        if SESSION is not None:
            SESSION.stop()
        sess = WebSession(
            model=HOLDER.model,
            vocab=HOLDER.vocab,
            device=HOLDER.device,
            seed_tokens=seed_tokens,
            scheduler=SCHEDULER,
            temp=float(data.get("temperature", 1.2)),
            top_p=float(data.get("top_p", 0.9)),
            chunk_size=int(data.get("chunk_size", 32)),
            max_len=int(data.get("max_len", 512)),
            pitch_min=int(data.get("pitch_min", 0)),
            pitch_max=int(data.get("pitch_max", 127)),
            lead_cap=float(data.get("lead_cap", 20.0)),
        )
        sess.start()
        SESSION = sess

    return jsonify({
        "ok": True,
        "seed": seed_path.stem,
        "seed_tokens": len(seed_tokens),
        "device": HOLDER.device,
    })


@app.route("/api/anchor", methods=["POST"])
def anchor():
    data = request.get_json(force=True)
    aid  = data.get("anchor_id")
    with SESSION_LOCK:
        if SESSION is None:
            return jsonify({"error": "no active session"}), 400
        try:
            path = _resolve("anchor", aid)
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        SESSION.feed_anchor(path.stem, str(path))
    return jsonify({"ok": True, "anchor": path.stem})


@app.route("/api/stop", methods=["POST"])
def stop():
    global SESSION
    with SESSION_LOCK:
        if SESSION is not None:
            SESSION.stop()
            SESSION = None
    return jsonify({"ok": True})


@app.route("/api/panic", methods=["POST"])
def panic():
    """All Sound Off + All Notes Off — 소리가 계속 날 때 긴급 정지."""
    global SESSION
    with SESSION_LOCK:
        if SESSION is not None:
            SESSION.stop()
            SESSION = None
    SCHEDULER.panic()
    return jsonify({"ok": True})


@app.route("/api/stream")
def stream():
    def gen():
        yield "retry: 2000\n\n"
        while True:
            sess = SESSION
            if sess is None:
                yield "event: idle\ndata: {}\n\n"
                time.sleep(0.5)
                continue
            events = sess.drain_sse(timeout=0.3)
            if events:
                yield "data: " + json.dumps(events) + "\n\n"
            else:
                yield ": keepalive\n\n"

    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


def main():
    global HOLDER, SCHEDULER
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="trained_model/model_ep325.pt")
    ap.add_argument("--melody-port", default="melody")
    ap.add_argument("--pad-port",    default="pad")
    ap.add_argument("--host",  default="127.0.0.1")
    ap.add_argument("--port",  type=int, default=5000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--pitch-min", type=int, default=0)
    ap.add_argument("--pitch-max", type=int, default=127)
    args = ap.parse_args()

    os.chdir(ROOT)  # 상대경로 기준을 프로젝트 루트로

    HOLDER = ModelHolder(args.model_path, device=args.device)

    SCHEDULER = MIDIScheduler(args.melody_port, args.pad_port)
    ok = SCHEDULER.open_ports()
    if not ok:
        print(f"[midi] WARNING: {SCHEDULER._port_error}", flush=True)
        print(f"[midi] Available ports: {MIDIScheduler.list_ports()}", flush=True)
        print("[midi] Continuing without MIDI output (piano roll still works).", flush=True)

    def _shutdown():
        print("\n[server] shutdown → sending All Sound Off ...", flush=True)
        global SESSION
        if SESSION is not None:
            SESSION.stop()
            SESSION = None
        SCHEDULER.panic()

    atexit.register(_shutdown)

    def _sig_handler(sig, frame):
        _shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT,  _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    print(f"\n  ThemeTransformer Web  →  http://{args.host}:{args.port}\n", flush=True)
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
