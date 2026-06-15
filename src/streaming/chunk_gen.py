"""Auto-regressive chunk generator thread for ThemeTransformer streaming."""
import threading
import torch
import numpy as np


class ChunkGenerator(threading.Thread):
    """Runs in background, producing token chunks and updating encoder theme.

    Generator loop:
    1. Check AnchorQueue – if a MIDI path is waiting, update input_theme (encoder).
    2. Generate `chunk_size` tokens auto-regressively.
    3. Push the chunk onto TokenChunkQueue for the MIDI output engine.
    """

    def __init__(self, model, vocab, theme_seq, device,
                 anchor_queue, token_queue,
                 chunk_size: int = 32,
                 max_len: int = 512,
                 temp: float = 1.2,
                 top_p: float = 0.9,
                 pitch_min: int = 0,
                 pitch_max: int = 127):
        super().__init__(daemon=True)
        self.model = model
        self.vocab = vocab
        self.device = device
        self.anchor_queue = anchor_queue
        self.token_queue = token_queue
        self.chunk_size = chunk_size
        self.max_len = max_len
        self.temp = temp
        self.top_p = top_p
        self._stop_event = threading.Event()

        self._theme_lock = threading.Lock()
        self.input_theme = torch.tensor(theme_seq).reshape(-1, 1).to(device)

        # decoder context state
        self.context: list[int] = []
        self.label_list: list[int] = []
        self.previous_labeled: bool = False

        # pitch masking
        self.forbidden_ids = [
            tid for tok, tid in vocab.token2id.items()
            if tok.startswith("Note-On")
            and not (pitch_min <= int(tok.split("_")[1]) <= pitch_max)
        ]

    # ── public API ────────────────────────────────────────────────────────────

    def init_seed(self, seed_tokens: list[int]):
        """Set the initial decoder context from seed MIDI tokens (literal)."""
        # Decoder always starts with Theme_Start as first token
        ts_id = self.vocab.token2id["Theme_Start"]
        self.context = [ts_id] + list(seed_tokens)

        # Build label list: Theme_Start opens the theme region (label increments),
        # but seed music tokens are outside the theme region so they get label=0.
        # We mark Theme_Start as label=1 and immediately close (previous_labeled=False)
        # so generated continuation stays unlabeled.
        self.label_list = [0] * len(self.context)
        prev = False
        for i, tok_id in enumerate(self.context):
            tok = self.vocab.id2token[tok_id]
            if tok == "Theme_Start":
                prev = True
            elif tok == "Theme_End":
                prev = False
            if prev:
                self.label_list[i] = 1 if i == 0 else self.label_list[i - 1] + 1
        self.previous_labeled = prev

    def update_theme(self, new_theme_seq: list[int]):
        """Thread-safe encoder theme update (called on anchor injection)."""
        new_tensor = torch.tensor(new_theme_seq).reshape(-1, 1).to(self.device)
        with self._theme_lock:
            self.input_theme = new_tensor

    def feed_anchor(self, midi_path: str):
        self.anchor_queue.put(midi_path)

    def stop(self):
        self._stop_event.set()

    # ── internal helpers ──────────────────────────────────────────────────────

    def _append_token(self, tok_id: int):
        tok = self.vocab.id2token[tok_id]
        if tok == "Theme_Start":
            self.previous_labeled = True
        elif tok == "Theme_End":
            self.previous_labeled = False
        label = (self.label_list[-1] + 1) if self.previous_labeled else 0
        self.context.append(tok_id)
        self.label_list.append(label)

    def _inject_anchor(self, midi_path: str):
        """Anchor updates the encoder theme (not the decoder context)."""
        anchor_tokens = self.vocab.midi2TSD(midi_path, theme_annotations=False)
        new_theme_seq = (
            [self.vocab.token2id["Theme_Start"]]
            + anchor_tokens
            + [self.vocab.token2id["Theme_End"]]
        )
        self.update_theme(new_theme_seq)

    def _sample_next(self) -> tuple[int, str] | None:
        """Run one forward pass and sample a token. Returns (tok_id, tok_str) or None."""
        ctx = self.context[-self.max_len:]
        lbl = self.label_list[-self.max_len:]

        input_x = torch.tensor(ctx).reshape(-1, 1).to(self.device)
        label_input = torch.tensor(lbl).reshape(-1, 1).to(self.device)
        att_msk = self.model.transformer_model.generate_square_subsequent_mask(
            input_x.shape[0]
        ).to(self.device)

        with self._theme_lock:
            cur_theme = self.input_theme

        with torch.no_grad():
            logits = self.model(
                src=cur_theme,
                tgt=input_x,
                tgt_label=label_input,
                tgt_mask=att_msk,
            )
            logits = torch.squeeze(logits[-1:]).cpu().numpy()

        if self.forbidden_ids:
            logits[self.forbidden_ids] = -1e9

        probs = self.model.temperature(logits=logits, temperature=self.temp)
        word = self.model.nucleus(probs=probs, p=self.top_p)

        if word == 0:
            return None

        w2e = self.vocab.id2token
        prev = w2e[self.context[-1]]
        cur = w2e[word]

        # grammar: Note-On → Note-Duration → Note-Velocity (same track)
        def _track(tok_str):
            return tok_str.split("_")[0].split("-")[2]

        if "Note-On" in prev and "Note-Duration" not in cur:
            return None
        if "Note-Duration" in cur and "Note-On" not in prev:
            return None
        if "Note-On" in prev and "Note-Duration" in cur:
            if _track(prev) != _track(cur):
                return None
        if "Note-Duration" in prev and "Note-Velocity" not in cur:
            return None
        if "Note-Velocity" in cur and "Note-Duration" not in prev:
            return None
        if "Note-Duration" in prev and "Note-Velocity" in cur:
            if _track(prev) != _track(cur):
                return None

        # Theme region consistency
        if cur.startswith("Theme"):
            if cur == "Theme_Start" and self.previous_labeled:
                return None
            if cur == "Theme_End" and not self.previous_labeled:
                return None

        return word, cur

    # ── main loop ─────────────────────────────────────────────────────────────

    def run(self):
        self.model.eval()

        while not self._stop_event.is_set():
            # 1. Anchor injection → updates encoder theme only (no decoder context change)
            anchor_path = self.anchor_queue.get_nowait()
            if anchor_path is not None:
                self._inject_anchor(anchor_path)
                self.token_queue.put(("anchor_signal", []))  # SSE marker용 신호
                continue

            # 2. Generate one chunk
            chunk: list[int] = []
            fail_cnt = 0
            while len(chunk) < self.chunk_size and not self._stop_event.is_set():
                result = self._sample_next()
                if result is None:
                    fail_cnt += 1
                    if fail_cnt > 512:
                        print("[ChunkGenerator] stuck – try a different seed/anchor")
                        self._stop_event.set()
                        break
                    continue
                tok_id, _ = result
                self._append_token(tok_id)
                chunk.append(tok_id)
                fail_cnt = 0

            if chunk:
                self.token_queue.put(("generated", chunk))
