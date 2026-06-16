"""Thread-safe queues for the streaming pipeline."""
import queue


class AnchorQueue:
    """MIDI file paths queued for literal injection into the decoder context."""

    def __init__(self):
        self._q = queue.Queue()

    def put(self, midi_path: str):
        self._q.put(midi_path)

    def get_nowait(self):
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def empty(self):
        return self._q.empty()


class TokenChunkQueue:
    """Chunks of token ids passed from generator to MIDI output engine.

    Each item is a tuple (kind, token_ids) where kind is 'generated' or 'anchor'.
    """

    def __init__(self, maxsize: int = 16):
        self._q = queue.Queue(maxsize=maxsize)

    def put(self, item, timeout=None):
        # timeout → raises queue.Full so the producer can stay responsive to a
        # stop signal instead of blocking forever on a full queue (thread leak).
        self._q.put(item, timeout=timeout)

    def get(self, timeout: float = 1.0):
        return self._q.get(timeout=timeout)

    def empty(self):
        return self._q.empty()
