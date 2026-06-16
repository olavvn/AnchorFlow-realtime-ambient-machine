"""ThemeStreamEngine – public API for continuous ambient MIDI generation."""
import sys
import os
import torch

# allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from preprocess.vocab import Vocab
from mymodel import myLM
from .buffer import AnchorQueue, TokenChunkQueue
from .chunk_gen import ChunkGenerator
from .midi_out import MIDIOutputEngine, list_output_ports


class ThemeStreamEngine:
    """Orchestrates theme conditioning, streaming generation, and dual-port MIDI output.

    Usage
    -----
    engine = ThemeStreamEngine(
        model_path="trained_model/model_ep325.pt",
        theme_midi="theme.mid",
        melody_port="melody",   # substring of loopMIDI port name
        pad_port="pad",
    )
    engine.set_seed("seed.mid")   # optional – literal starting passage
    engine.start()

    # later, feed an anchor to steer the music
    engine.feed_anchor("anchor1.mid")

    engine.stop()
    """

    def __init__(
        self,
        model_path: str,
        theme_midi: str,
        melody_port: str = "melody",
        pad_port: str = "pad",
        chunk_size: int = 32,
        max_len: int = 512,
        temp: float = 1.2,
        top_p: float = 0.9,
        pitch_min: int = 0,
        pitch_max: int = 127,
        cuda: bool | None = None,
        latency_s: float = 0.0,
    ):
        self.vocab = Vocab()

        # device
        if cuda is None:
            cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if cuda else "cpu")

        # model
        self.model = myLM(
            self.vocab.n_tokens,
            d_model=256,
            num_encoder_layers=6,
            xorpattern=[0, 0, 0, 1, 1, 1],
        )
        print(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"Model ready on {self.device}")

        # theme → encoder input
        theme_tokens = self.vocab.midi2TSD(theme_midi, theme_annotations=False)
        self.theme_seq = (
            [self.vocab.token2id["Theme_Start"]]
            + theme_tokens
            + [self.vocab.token2id["Theme_End"]]
        )
        print(f"Theme loaded: {len(self.theme_seq)} tokens from '{theme_midi}'")

        # queues
        self._anchor_queue = AnchorQueue()
        self._token_queue  = TokenChunkQueue(maxsize=16)

        # generator
        self._generator = ChunkGenerator(
            model=self.model,
            vocab=self.vocab,
            theme_seq=self.theme_seq,
            device=self.device,
            anchor_queue=self._anchor_queue,
            token_queue=self._token_queue,
            chunk_size=chunk_size,
            max_len=max_len,
            temp=temp,
            top_p=top_p,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
        )

        # MIDI output
        self._midi_engine = MIDIOutputEngine(
            token_queue=self._token_queue,
            vocab=self.vocab,
            melody_port=melody_port,
            pad_port=pad_port,
            latency_s=latency_s,
        )

        self._seed_midi: str | None = None

    # ── configuration ─────────────────────────────────────────────────────────

    def set_seed(self, midi_path: str):
        """Set a seed MIDI that will be played literally at the start of the stream."""
        self._seed_midi = midi_path

    def feed_anchor(self, midi_path: str):
        """Queue an anchor MIDI for literal injection into the running stream."""
        self._anchor_queue.put(midi_path)
        print(f"[engine] anchor queued: {midi_path}")

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        """Start the MIDI output engine and the generator thread."""
        # initialise generator context
        if self._seed_midi is not None:
            seed_tokens = self.vocab.midi2TSD(self._seed_midi, theme_annotations=False)
            self._generator.init_seed(seed_tokens)
            # push seed tokens so MIDI engine plays them immediately
            if seed_tokens:
                self._token_queue.put(("anchor", seed_tokens))
            print(f"[engine] seed loaded: {len(seed_tokens)} tokens from '{self._seed_midi}'")
        else:
            # minimal context: just Theme_Start
            self._generator.init_seed([])

        self._midi_engine.start()
        self._generator.start()
        print("[engine] streaming started – press Ctrl+C or call stop() to end")

    def stop(self):
        self._generator.stop()
        self._midi_engine.stop()
        self._generator.join(timeout=2)
        self._midi_engine.join(timeout=2)
        print("[engine] stopped")

    @staticmethod
    def list_ports():
        """Print available MIDI output ports."""
        ports = list_output_ports()
        if ports:
            for i, name in enumerate(ports):
                print(f"  [{i}] {name}")
        else:
            print("  (no MIDI output ports found)")
        return ports
