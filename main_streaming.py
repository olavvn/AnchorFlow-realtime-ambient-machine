"""ThemeTransformer Streaming CLI

Continuous ambient MIDI generation conditioned on a theme.
Outputs MELODY and PAD to separate loopMIDI virtual ports in real-time.

Usage examples
--------------
# List available MIDI ports
python main_streaming.py --list-ports

# Start streaming (theme required; seed and anchor are optional)
python main_streaming.py \\
    --theme theme_files/example_theme.mid \\
    --seed  seed_files/my_seed.mid \\
    --melody-port "melody" \\
    --pad-port    "pad"

Interactive commands while running
-----------------------------------
  a <path>   Feed an anchor MIDI (e.g.  a anchors/chord_change.mid)
  q          Quit
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.streaming.engine import ThemeStreamEngine


def parse_args():
    p = argparse.ArgumentParser(
        description="ThemeTransformer continuous MIDI streaming to loopMIDI"
    )
    p.add_argument("--model-path", default="trained_model/model_ep325.pt",
                   help="Path to model checkpoint")
    p.add_argument("--theme", required=False, default=None,
                   help="Theme MIDI file (MELODY=prog0, PAD=prog88)")
    p.add_argument("--seed", default=None,
                   help="Seed MIDI file – played literally at stream start")
    p.add_argument("--melody-port", default="melody",
                   help="loopMIDI port name substring for MELODY track")
    p.add_argument("--pad-port", default="pad",
                   help="loopMIDI port name substring for PAD track")
    p.add_argument("--chunk-size", type=int, default=32,
                   help="Tokens generated per forward-pass batch")
    p.add_argument("--max-len", type=int, default=512,
                   help="Decoder context window (tokens)")
    p.add_argument("--temp", type=float, default=1.2,
                   help="Sampling temperature")
    p.add_argument("--top-p", type=float, default=0.9,
                   help="Nucleus sampling p")
    p.add_argument("--pitch-min", type=int, default=0)
    p.add_argument("--pitch-max", type=int, default=127)
    p.add_argument("--cuda", action="store_true",
                   help="Force CUDA (auto-detected if omitted)")
    p.add_argument("--latency", type=float, default=0.0,
                   help="Extra output latency in seconds (use to compensate DAW buffer)")
    p.add_argument("--list-ports", action="store_true",
                   help="List available MIDI output ports and exit")
    return p.parse_args()


def main():
    args = parse_args()

    if args.list_ports:
        print("Available MIDI output ports:")
        ThemeStreamEngine.list_ports()
        return

    if args.theme is None:
        print("Error: --theme is required.\nRun with --list-ports to see available MIDI ports.")
        sys.exit(1)

    engine = ThemeStreamEngine(
        model_path=args.model_path,
        theme_midi=args.theme,
        melody_port=args.melody_port,
        pad_port=args.pad_port,
        chunk_size=args.chunk_size,
        max_len=args.max_len,
        temp=args.temp,
        top_p=args.top_p,
        pitch_min=args.pitch_min,
        pitch_max=args.pitch_max,
        cuda=args.cuda or None,
        latency_s=args.latency,
    )

    if args.seed:
        engine.set_seed(args.seed)

    engine.start()

    print("\nStreaming… Type  'a <midi_path>'  to feed an anchor, 'q' to quit.\n")
    try:
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                continue
            if line.lower() in ("q", "quit", "exit"):
                break
            if line.lower().startswith("a "):
                path = line[2:].strip()
                if os.path.isfile(path):
                    engine.feed_anchor(path)
                else:
                    print(f"File not found: {path}")
            else:
                print("Commands:  a <midi_path>  |  q")
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
