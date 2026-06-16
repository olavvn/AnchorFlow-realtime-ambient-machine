"""Auto-regressive chunk generator thread for ThemeTransformer streaming."""
import threading
import traceback
import queue
import torch
import numpy as np


class ChunkGenerator(threading.Thread):
    """Runs in background, producing token chunks and updating encoder theme.

    Generator loop:
    1. Check AnchorQueue – if a MIDI path is waiting, insert it as a new theme
       region (literal decoder context + new encoder theme).
    2. Generate `chunk_size` tokens auto-regressively.
    3. Push the chunk onto TokenChunkQueue for the MIDI output engine.

    Why this is fast enough AND musically bounded:
    - The encoder memory is cached and only recomputed when the theme changes,
      saving a full encoder pass on every token.
    - Grammar is enforced by *masking the logits* before sampling, so every
      forward pass yields a valid token (no wasted reject-resample forwards).
    - Density caps keep the *musical* rate above real-time. Musical time only
      advances on Time-Shift tokens; left unconstrained the model stacks dense
      chords (no time advance → starves the clock → underrun/desync) and emits
      runaway Time-Shift runs (multi-second silent gaps). The caps below bound
      both: max single shift, max consecutive notes (force a shift), and max
      consecutive shifts (force a note).
    """

    def __init__(self, model, vocab, theme_seq, device,
                 anchor_queue, token_queue,
                 chunk_size: int = 24,
                 max_len: int = 512,
                 temp: float = 1.0,
                 top_p: float = 0.9,
                 time_scale: float = 1.0,
                 theme_recur_sec: float = 0.0,
                 theme_recur_mode: str = "auto",
                 pitch_min: int = 0,
                 pitch_max: int = 127,
                 max_timeshift: int = 25,
                 min_force_shift: int = 12,
                 max_consec_notes: int = 4,
                 max_consec_shifts: int = 2):
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
        self.max_consec_notes = max_consec_notes
        self.max_consec_shifts = max_consec_shifts
        self.min_force_shift = min_force_shift
        self._stop_event = threading.Event()

        # encoder theme + cached memory (recomputed only when theme changes)
        self._theme_lock = threading.Lock()
        self.input_theme = torch.tensor(theme_seq).reshape(-1, 1).to(device)
        self._theme_ver = 0          # bumped on every theme replacement
        self._mem = None             # cached encoder memory
        self._mem_ver = -1           # version the cached memory was built for
        self._causal_mask = None     # cached causal attention mask (built once)

        # decoder context state
        self.context: list[int] = []
        self.label_list: list[int] = []
        self.previous_labeled: bool = False

        # density-cap counters
        self._notes_since_shift = 0
        self._consec_shifts = 0

        # periodic theme recurrence — keep the current theme's color present by
        # re-inserting a labeled [Theme_Start, *theme, Theme_End] region every
        # `theme_recur_sec` seconds of generated music (0 = disabled).
        self.time_scale = time_scale
        self.theme_recur_sec = theme_recur_sec
        self.theme_recur_mode = theme_recur_mode
        self._ts_id = self.vocab.token2id["Theme_Start"]
        self._te_id = self.vocab.token2id["Theme_End"]
        self._current_theme_tokens: list[int] = []
        self._music_sec = 0.0        # generated musical seconds (time_scale applied)
        self._last_theme_sec = 0.0   # _music_sec at the last theme presence

        self._build_grammar_masks(pitch_min, pitch_max,
                                  min_force_shift, max_timeshift)

    # ── grammar / logit masks ──────────────────────────────────────────────────

    def _build_grammar_masks(self, pitch_min, pitch_max, min_timeshift, max_timeshift):
        n = self.vocab.n_tokens
        t2i = self.vocab.token2id
        tracks = self.vocab.tracks   # ["MELODY", "PAD"]

        def block(prefix, allowed_fn=lambda tok: True):
            return np.array([tid for tok, tid in t2i.items()
                             if tok.startswith(prefix) and allowed_fn(tok)],
                            dtype=np.int64)

        # Time-Shift restricted to [min_timeshift, max_timeshift] steps. The lower
        # bound is the real-time guarantee: without it the model shifts by 1-2
        # steps (~0.08s) while stacking dense chords, so music time barely
        # advances per token → generation collapses far below real-time. Forcing a
        # minimum shift means every note group advances music enough to keep up.
        ts_ids = block("Time-Shift_",
                       lambda t: min_timeshift <= int(t.split("_")[1]) <= max_timeshift)

        def pitch_ok(tok):
            return pitch_min <= int(tok.split("_")[1]) <= pitch_max
        on_ids = {tr: block(f"Note-On-{tr}_", pitch_ok) for tr in tracks}
        dur_ids = {tr: block(f"Note-Duration-{tr}_") for tr in tracks}
        vel_ids = {tr: block(f"Note-Velocity-{tr}_") for tr in tracks}

        all_on = np.concatenate([on_ids[tr] for tr in tracks])

        def mk(*id_arrays):
            m = np.zeros(n, dtype=bool)
            for a in id_arrays:
                m[a] = True
            return m

        # "free" state variants (after Note-Velocity / Time-Shift / boundary)
        self._mask_free_both = mk(all_on, ts_ids)   # note OR shift
        self._mask_free_note = mk(all_on)           # force a note (too many shifts)
        self._mask_free_shift = mk(ts_ids)          # force a shift (too many notes)

        self._mask_dur = {tr: mk(dur_ids[tr]) for tr in tracks}
        self._mask_vel = {tr: mk(vel_ids[tr]) for tr in tracks}

        # token id → (category, track)
        self._cat = {}
        for tok, tid in t2i.items():
            if tok.startswith("Note-On-"):
                self._cat[tid] = ("on", tok.split("_")[0].split("-")[2])
            elif tok.startswith("Note-Duration-"):
                self._cat[tid] = ("dur", tok.split("_")[0].split("-")[2])
            elif tok.startswith("Note-Velocity-"):
                self._cat[tid] = ("vel", tok.split("_")[0].split("-")[2])
            elif tok.startswith("Time-Shift"):
                self._cat[tid] = ("shift", None)
            else:
                self._cat[tid] = ("other", None)

    def _allow_mask(self) -> np.ndarray:
        """Allow-mask for the next token from the last token + density counters."""
        if not self.context:
            return self._mask_free_both
        cat, track = self._cat.get(self.context[-1], ("other", None))
        if cat == "on":
            return self._mask_dur[track]
        if cat == "dur":
            return self._mask_vel[track]

        # free state: apply density caps (mutually exclusive triggers)
        if self._consec_shifts >= self.max_consec_shifts:
            base_mask = self._mask_free_note
        elif self._notes_since_shift >= self.max_consec_notes:
            base_mask = self._mask_free_shift
        else:
            base_mask = self._mask_free_both

        if self.theme_recur_mode == "auto":
            mask = base_mask.copy()
            if self.previous_labeled:
                mask[self._te_id] = True
            else:
                mask[self._ts_id] = True
            return mask

        return base_mask

    # ── public API ────────────────────────────────────────────────────────────

    def init_seed(self, seed_tokens: list[int]):
        """Initial decoder context: [Theme_Start, *seed, Theme_End], labels
        1..N+1 then 0 (training format). Subsequent generation is label 0."""
        ts_id = self.vocab.token2id["Theme_Start"]
        te_id = self.vocab.token2id["Theme_End"]
        self.context = [ts_id] + list(seed_tokens) + [te_id]

        self.label_list = []
        prev = False
        cnt = 0
        for tok_id in self.context:
            tok = self.vocab.id2token[tok_id]
            if tok == "Theme_Start":
                prev = True
            if tok == "Theme_End":
                prev = False
            if prev:
                cnt += 1
                self.label_list.append(cnt)
            else:
                self.label_list.append(0)
        self.previous_labeled = prev

        # remember this theme for periodic recurrence; restart the recur timer
        self._current_theme_tokens = list(seed_tokens)
        self._last_theme_sec = self._music_sec

    def update_theme(self, new_theme_seq: list[int]):
        new_tensor = torch.tensor(new_theme_seq).reshape(-1, 1).to(self.device)
        with self._theme_lock:
            self.input_theme = new_tensor
            self._theme_ver += 1

    def feed_anchor(self, item):
        self.anchor_queue.put(item)

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

        # density counters + generated musical-time clock
        cat, _ = self._cat.get(tok_id, ("other", None))
        if cat == "on":
            self._notes_since_shift += 1
            self._consec_shifts = 0
        elif cat == "shift":
            self._consec_shifts += 1
            self._notes_since_shift = 0
            steps = int(tok.split("_")[1])
            self._music_sec += steps * self.vocab.time_resolution * self.time_scale

    def _inject_anchor(self, midi_path: str) -> list[int]:
        """Anchor → new encoder theme AND a *fresh* decoder context seeded with
        the anchor as a labeled theme region.

        Cross-attention to the encoder memory is gated to label>0 positions
        (myTransformer.py:567-573), so generated (label-0) tokens are
        conditioned on the theme ONLY indirectly, via self-attention over the
        decoder context. If we *appended* the anchor to the existing context,
        ~500 tokens of the previous theme's development would keep dominating
        self-attention and the new theme would be forgotten within seconds
        ("anchor briefly appears then disappears"). So we *reset* the context to
        the anchor region — the model then develops the new theme from scratch,
        exactly like the initial seed, making the anchor the new dominant theme.
        """
        anchor_tokens = self.vocab.midi2TSD(midi_path, theme_annotations=False)
        if not anchor_tokens:
            print(f"[ChunkGenerator] anchor '{midi_path}' produced 0 tokens; ignored",
                  flush=True)
            return []
        ts_id = self.vocab.token2id["Theme_Start"]
        te_id = self.vocab.token2id["Theme_End"]

        self.update_theme([ts_id] + anchor_tokens + [te_id])

        # Anchor becomes the new seed: reset decoder context (labels 1..N then 0)
        # and density counters so generation develops from the new theme alone.
        # init_seed also stores anchor_tokens as the current theme + restarts the
        # recurrence timer, so periodic re-injection now uses this anchor.
        self.init_seed(anchor_tokens)
        self._notes_since_shift = 0
        self._consec_shifts = 0
        return anchor_tokens

    def _reinject_theme(self) -> list[int]:
        """Re-insert the current theme as a labeled [Theme_Start, *theme,
        Theme_End] region into the decoder context AND play it literally,
        WITHOUT resetting context. Follows the seed/anchor principle that an
        inserted theme region is heard.

        This both (a) re-grounds cross-attention (gated to label>0) on the theme
        so label-0 generation keeps developing in-theme instead of drifting, and
        (b) audibly restates the motif. The caller emits the returned tokens to
        the output queue so they are scheduled/played and drawn on the roll."""
        toks = self._current_theme_tokens
        if not toks:
            return []
        ts_id = self.vocab.token2id["Theme_Start"]
        te_id = self.vocab.token2id["Theme_End"]
        self._append_token(ts_id)
        for tok_id in toks:
            self._append_token(tok_id)
        self._append_token(te_id)
        self._last_theme_sec = self._music_sec
        self._notes_since_shift = 0
        self._consec_shifts = 0
        return toks

    def _ensure_memory(self):
        with self._theme_lock:
            theme = self.input_theme
            ver = self._theme_ver
        if ver != self._mem_ver or self._mem is None:
            with torch.no_grad():
                self._mem = self.model.encode_theme(theme)
            self._mem_ver = ver
        return self._mem

    def _sample_logits(self, logits: np.ndarray, allow: np.ndarray) -> int:
        """Temperature + nucleus (top-p) sampling restricted to `allow`."""
        logits = np.where(allow, logits, -np.inf)
        logits = logits / max(self.temp, 1e-6)
        logits -= np.nanmax(logits)
        probs = np.exp(logits)
        probs[~np.isfinite(probs)] = 0.0
        s = probs.sum()
        if s <= 0:                       # safety: uniform over allowed
            idx = np.flatnonzero(allow)
            return int(np.random.choice(idx))
        probs /= s

        # proper top-p truncation
        order = np.argsort(probs)[::-1]
        csum = np.cumsum(probs[order])
        keep = np.searchsorted(csum, self.top_p) + 1
        keep = max(1, min(keep, len(order)))
        cand = order[:keep]
        cp = probs[cand]
        cp /= cp.sum()
        return int(np.random.choice(cand, p=cp))

    def _sample_next(self) -> int:
        ctx = self.context[-self.max_len:]
        lbl = self.label_list[-self.max_len:]

        n = len(ctx)
        input_x = torch.tensor(ctx, device=self.device).reshape(-1, 1)
        label_input = torch.tensor(lbl, device=self.device).reshape(-1, 1)
        # Causal mask depends only on length; build once (sized max_len) and
        # slice. Rebuilding a max_len×max_len tensor + host→GPU copy every token
        # was pure per-token overhead (generation is GIL/overhead-bound).
        if self._causal_mask is None or self._causal_mask.shape[0] < n:
            self._causal_mask = (
                self.model.transformer_model.generate_square_subsequent_mask(
                    max(n, self.max_len)).to(self.device))
        att_msk = self._causal_mask[:n, :n]

        memory = self._ensure_memory()
        with torch.no_grad():
            logits = self.model.decode_step(
                tgt=input_x, memory=memory,
                tgt_label=label_input, tgt_mask=att_msk,
            )
            logits = torch.squeeze(logits[-1:]).float().cpu().numpy()

        return self._sample_logits(logits, self._allow_mask())

    # ── main loop ─────────────────────────────────────────────────────────────

    def _emit(self, item) -> bool:
        """Put a chunk on the output queue but stay responsive to stop(): if the
        queue is full (reader paused/stopped) we retry with a timeout and bail
        when the stop event is set, instead of blocking forever (thread leak)."""
        while not self._stop_event.is_set():
            try:
                self.token_queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def run(self):
        self.model.eval()
        print(f"[gen] thread START id={threading.get_ident()}", flush=True)
        while not self._stop_event.is_set():
            try:
                self._run_once()
            except Exception:
                # never let a transient error kill the stream
                print("[ChunkGenerator] error (continuing):", flush=True)
                traceback.print_exc()
        print(f"[gen] thread EXIT id={threading.get_ident()}", flush=True)

    def _run_once(self):
        # 1. Anchor: labeled-theme insertion + encoder theme update
        anchor_item = self.anchor_queue.get_nowait()
        if anchor_item is not None:
            label, anchor_path = anchor_item
            anchor_tokens = self._inject_anchor(anchor_path)
            if anchor_tokens:
                print(f"[ChunkGenerator] anchor '{label}' injected "
                      f"({len(anchor_tokens)} tokens)", flush=True)
                self._emit(("anchor", label, anchor_tokens))
            return

        # 1b. Periodic theme recurrence: literally restate the current theme
        # (played + re-grounds conditioning) so its color persists until the next
        # anchor, per the seed/anchor "inserted theme is heard" principle.
        if (self.theme_recur_mode == "interval" and self.theme_recur_sec > 0 and self._current_theme_tokens
                and (self._music_sec - self._last_theme_sec) >= self.theme_recur_sec):
            toks = self._reinject_theme()
            if toks:
                self._emit(("theme", "THEME", toks))
            return

        # 2. Generate one chunk (masked sampling → always valid)
        chunk: list[int] = []
        while len(chunk) < self.chunk_size and not self._stop_event.is_set():
            if not self.anchor_queue.empty():
                break   # handle a pending anchor promptly
            if (self.theme_recur_mode == "interval" and self.theme_recur_sec > 0 and self._current_theme_tokens
                    and (self._music_sec - self._last_theme_sec) >= self.theme_recur_sec):
                break   # theme recurrence is due → flush chunk, re-inject next pass
            tok_id = self._sample_next()
            self._append_token(tok_id)
            chunk.append(tok_id)

        if chunk:
            self._emit(("generated", None, chunk))
