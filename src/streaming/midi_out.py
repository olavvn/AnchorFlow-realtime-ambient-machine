"""Real-time MIDI output engine – routes MELODY and PAD to separate loopMIDI ports."""
import queue
import threading
import time
from typing import Optional


def list_output_ports() -> list[str]:
    """Return available MIDI output port names."""
    try:
        import rtmidi
        out = rtmidi.MidiOut()
        return out.get_ports()
    except Exception as e:
        print(f"[midi_out] rtmidi unavailable: {e}")
        return []


def _open_port(port_name: str, label: str):
    """Open an rtmidi output port by name substring match. Returns port object."""
    import rtmidi
    out = rtmidi.MidiOut()
    ports = out.get_ports()
    for i, name in enumerate(ports):
        if port_name.lower() in name.lower():
            out.open_port(i)
            print(f"[midi_out] {label} → port {i}: {name}")
            return out
    raise RuntimeError(
        f"MIDI port '{port_name}' not found.\n"
        f"Available ports: {ports}\n"
        "Create the port in loopMIDI and try again."
    )


class MIDIOutputEngine(threading.Thread):
    """Consumes token chunks and plays them in real-time via two loopMIDI ports.

    MELODY tokens → melody_port
    PAD tokens    → pad_port

    Time-Shift tokens advance an internal stream clock, and the thread sleeps
    to keep wall-clock time aligned with music time.
    """

    # MIDI channel 0 for both tracks
    _NOTE_ON  = 0x90
    _NOTE_OFF = 0x80

    def __init__(
        self,
        token_queue,
        vocab,
        melody_port: str = "melody",
        pad_port: str = "pad",
        latency_s: float = 0.0,
    ):
        super().__init__(daemon=True)
        self.token_queue = token_queue
        self.vocab = vocab
        self.latency_s = latency_s
        self._melody_port_name = melody_port
        self._pad_port_name = pad_port

        self._stop_event = threading.Event()
        self._melody_out = None
        self._pad_out = None

        # stream clock
        self._start_wall: Optional[float] = None
        self._stream_time: float = 0.0

        # pending note state machine (completes on Note-Velocity)
        self._pending: Optional[dict] = None

        # note-off scheduler
        self._noteoff_queue: queue.PriorityQueue = queue.PriorityQueue()
        self._noteoff_thread: Optional[threading.Thread] = None

    # ── public API ────────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()
        # send all-notes-off on both channels
        try:
            if self._melody_out:
                self._melody_out.send_message([0xB0, 123, 0])
            if self._pad_out:
                self._pad_out.send_message([0xB0, 123, 0])
        except Exception:
            pass

    # ── internal ─────────────────────────────────────────────────────────────

    def _get_port(self, track: str):
        return self._melody_out if track == "MELODY" else self._pad_out

    def _schedule_noteoff(self, port, pitch: int, wall_time: float):
        self._noteoff_queue.put((wall_time, pitch, port))

    def _noteoff_loop(self):
        """Dedicated thread that fires note-off messages at the right moment."""
        while not self._stop_event.is_set():
            try:
                fire_at, pitch, port = self._noteoff_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            now = time.perf_counter()
            if fire_at > now:
                time.sleep(fire_at - now)
            if not self._stop_event.is_set():
                try:
                    port.send_message([self._NOTE_OFF, pitch, 0])
                except Exception:
                    pass

    def _process_token(self, tok_id: int):
        tok = self.vocab.id2token[tok_id]

        if tok.startswith("Time-Shift"):
            steps = int(tok.split("_")[1])
            self._stream_time += steps * self.vocab.time_resolution
            target = self._start_wall + self._stream_time + self.latency_s
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)

        elif tok.startswith("Note-On"):
            parts = tok.split("_")
            track = parts[0].split("-")[2]   # MELODY or PAD
            pitch = int(parts[1])
            self._pending = {"track": track, "pitch": pitch, "dur": None}

        elif tok.startswith("Note-Duration") and self._pending is not None:
            steps = int(tok.split("_")[1])
            self._pending["dur"] = steps * self.vocab.time_resolution

        elif tok.startswith("Note-Velocity") and self._pending is not None and self._pending["dur"] is not None:
            vel = int(tok.split("_")[1])
            track = self._pending["track"]
            pitch = self._pending["pitch"]
            dur   = self._pending["dur"]
            port  = self._get_port(track)

            if port is not None:
                try:
                    port.send_message([self._NOTE_ON, pitch, vel])
                    noteoff_wall = self._start_wall + self._stream_time + dur + self.latency_s
                    self._schedule_noteoff(port, pitch, noteoff_wall)
                except Exception as e:
                    print(f"[midi_out] send error: {e}")
            self._pending = None

        # Theme_Start / Theme_End / padding – silently skip

    def run(self):
        # open MIDI ports
        self._melody_out = _open_port(self._melody_port_name, "MELODY")
        self._pad_out    = _open_port(self._pad_port_name,    "PAD")

        # start note-off scheduler
        self._noteoff_thread = threading.Thread(target=self._noteoff_loop, daemon=True)
        self._noteoff_thread.start()

        self._start_wall = time.perf_counter()
        self._stream_time = 0.0

        while not self._stop_event.is_set():
            try:
                kind, tokens = self.token_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            for tok_id in tokens:
                if self._stop_event.is_set():
                    break
                self._process_token(tok_id)

        # cleanup
        if self._melody_out:
            self._melody_out.close_port()
        if self._pad_out:
            self._pad_out.close_port()
