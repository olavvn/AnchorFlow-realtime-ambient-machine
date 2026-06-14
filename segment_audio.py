"""Interactively split long tracks into segments by hand.

Second step of the workflow (after download_youtube_long.py):
    - reads every audio file in  <long>/
    - opens each one in your default media player so you can listen & scrub
    - you type the split points (e.g. `1:30 3:05 4:40`)
    - the track is cut at those points and the pieces are written to
          <out>/<title>/<title>-000.wav
          <out>/<title>/<title>-001.wav
          ...

Split points are *boundaries*: typing `30 60 90` on a 2-minute track yields
[0-30], [30-60], [60-90], [90-end] -> 4 segments. Times may be given as plain
seconds (`90`, `90.5`) or as mm:ss / hh:mm:ss (`1:30`, `1:02:03`).

Batch modes (no listening / prompts):
    --stamp FILE     read FILE where each line holds the split points for one track.
                     Line N maps to track N (tracks are sorted by name, so the
                     001-/002-/... prefixes from download_youtube_long.py line up).
                     A blank line means "don't split that track". Example file:
                         1:30 5:40 9:30
                         2:40 5:33 11:22
                         4:33 7:38 9:22
    --interval SEC   split every track into fixed-length SEC-second segments.
                     The final segment is kept even if shorter than SEC.

Requires `ffmpeg` and `ffprobe` on PATH.

Usage:
    python segment_audio.py                          # interactive
    python segment_audio.py --stamp timestamp.txt    # one line of splits per track
    python segment_audio.py --interval 70            # every 70 seconds
    python segment_audio.py --long "D:/in" --out "D:/out" --no-play
"""
import argparse
import os
import subprocess
import sys

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

DEFAULT_LONG = r"C:\Users\Changhyun Kim\Desktop\SG_Lecture\2026-01\DL for Music & Audio\datasets\ambient_youtube\long"
DEFAULT_OUT = r"C:\Users\Changhyun Kim\Desktop\SG_Lecture\2026-01\DL for Music & Audio\datasets\ambient_youtube"

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg", ".aac")


def duration_of(path):
    """Return track length in seconds via ffprobe (or None on failure)."""
    res = subprocess.run(
        [FFPROBE, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True)
    try:
        return float(res.stdout.strip())
    except (ValueError, AttributeError):
        return None


def fmt(seconds):
    """Seconds -> H:MM:SS.s for display."""
    if seconds is None:
        return "?"
    s = float(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return "{}:{:02d}:{:04.1f}".format(int(h), int(m), s)
    return "{}:{:04.1f}".format(int(m), s)


def parse_time(token):
    """Parse '90', '90.5', '1:30', '1:02:03' -> seconds (float). Raises ValueError."""
    token = token.strip()
    if not token:
        raise ValueError("empty time")
    parts = token.split(":")
    if len(parts) > 3:
        raise ValueError("too many ':' in '{}'".format(token))
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def parse_points(text):
    """Parse a line of split points separated by spaces/commas -> sorted unique list."""
    raw = text.replace(",", " ").split()
    pts = sorted({round(parse_time(tok), 3) for tok in raw})
    return pts


def open_in_player(path):
    """Open the file in the OS default media player (non-blocking)."""
    try:
        if sys.platform == "win32":
            os.startfile(path)  # noqa: S606  (intended: open in default app)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except OSError as e:
        print("  [could not open player] {}".format(e), file=sys.stderr)


def cut_segments(src, out_dir, base, ext, boundaries):
    """Cut `src` at the given segment boundaries (list of (start, end) seconds)."""
    os.makedirs(out_dir, exist_ok=True)
    made = 0
    for i, (start, end) in enumerate(boundaries):
        out_path = os.path.join(out_dir, "{}-{:03d}{}".format(base, i, ext))
        cmd = [
            FFMPEG, "-hide_banner", "-loglevel", "warning", "-y",
            "-i", src,
            "-ss", "{:.3f}".format(start),
        ]
        if end is not None:
            cmd += ["-to", "{:.3f}".format(end)]
        cmd += ["-c", "copy", out_path]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            # Stream copy can fail on some codecs at arbitrary cut points -> re-encode.
            cmd[cmd.index("copy")] = ext_codec(ext)
            rc = subprocess.run(cmd).returncode
        if rc == 0:
            made += 1
        else:
            print("  [cut failed] segment {}".format(i), file=sys.stderr)
    return made


def ext_codec(ext):
    """Fallback encoder for re-cutting when stream copy fails."""
    return {".wav": "pcm_s16le", ".flac": "flac", ".mp3": "libmp3lame"}.get(ext, "aac")


def boundaries_from_points(points, total):
    """Turn split points into (start, end) segment pairs covering [0, total]."""
    edges = [0.0] + [p for p in points if 0.0 < p < (total or float("inf"))] + [total]
    return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]


def points_from_interval(total, interval):
    """Split points at every `interval` seconds. The last (short) segment is kept."""
    if not total:
        return []
    pts = []
    t = interval
    while t < total:
        pts.append(round(t, 3))
        t += interval
    return pts


def cut_track(path, out_root, points):
    """Cut one track at the given split points; returns number of segments written."""
    base, ext = os.path.splitext(os.path.basename(path))
    out_dir = os.path.join(out_root, base)
    dur = duration_of(path)
    bounds = boundaries_from_points(points, dur)
    n = cut_segments(path, out_dir, base, ext, bounds)
    print("  {} -> {} segment(s) in {}".format(os.path.basename(path), n, out_dir))
    return n


def read_stamp_file(path):
    """Read a stamp file: one line of split points per track (blank line = no split)."""
    lines = []
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            lines.append(ln.strip())
    return lines


def run_stamp_mode(tracks, out_root, stamp_path):
    """Apply line N of the stamp file as split points for track N (1-based)."""
    lines = read_stamp_file(stamp_path)
    if len(lines) != len(tracks):
        print("Warning: {} stamp line(s) but {} track(s); pairing the first {}.".format(
            len(lines), len(tracks), min(len(lines), len(tracks))))
    done = 0
    for i, path in enumerate(tracks):
        if i >= len(lines):
            print("  [no stamp line] {} -> skipped".format(os.path.basename(path)))
            continue
        line = lines[i]
        if not line:
            print("  [blank line] {} -> skipped".format(os.path.basename(path)))
            continue
        try:
            points = parse_points(line)
        except ValueError as e:
            print("  [bad line {}] {} ({})".format(i + 1, line, e), file=sys.stderr)
            continue
        cut_track(path, out_root, points)
        done += 1
    return done


def run_interval_mode(tracks, out_root, interval):
    """Split every track into fixed-length segments of `interval` seconds."""
    done = 0
    for path in tracks:
        dur = duration_of(path)
        cut_track(path, out_root, points_from_interval(dur, interval))
        done += 1
    return done


def process_track(path, out_root, play):
    base, ext = os.path.splitext(os.path.basename(path))
    out_dir = os.path.join(out_root, base)
    dur = duration_of(path)

    print("\n" + "=" * 70)
    print("Track : {}".format(os.path.basename(path)))
    print("Length: {}".format(fmt(dur)))
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print("Note  : output folder already has files -> they will be overwritten if you proceed.")

    if play:
        open_in_player(path)

    while True:
        print("\nEnter split points (seconds or mm:ss, space/comma separated).")
        print("Commands:  [p]lay again   [s]kip   [q]uit   (empty = skip)")
        try:
            text = input("> ").strip()
        except EOFError:
            return "quit"

        low = text.lower()
        if low in ("q", "quit"):
            return "quit"
        if low in ("p", "play"):
            open_in_player(path)
            continue
        if low in ("", "s", "skip"):
            print("Skipped.")
            return "skip"

        try:
            points = parse_points(text)
        except ValueError as e:
            print("  [bad input] {}".format(e))
            continue

        bounds = boundaries_from_points(points, dur)
        print("\nWill create {} segment(s):".format(len(bounds)))
        for i, (a, b) in enumerate(bounds):
            print("  {:03d}: {} -> {}".format(i, fmt(a), fmt(b)))
        ok = input("Proceed? [Y/n] ").strip().lower()
        if ok in ("n", "no"):
            continue

        n = cut_segments(path, out_dir, base, ext, bounds)
        print("Saved {} segment(s) -> {}".format(n, out_dir))
        return "done"


def main():
    ap = argparse.ArgumentParser(
        description="Listen to long tracks and split them into segments by hand.")
    ap.add_argument("-l", "--long", default=DEFAULT_LONG, help="folder with the long source tracks")
    ap.add_argument("-o", "--out", default=DEFAULT_OUT, help="folder to write per-track segment folders")
    ap.add_argument("--no-play", action="store_true", help="don't auto-open tracks in the media player")
    ap.add_argument("--stamp", metavar="FILE",
                    help="batch mode: text file with one line of split points per track "
                         "(line N -> track N, blank line = no split). No prompts.")
    ap.add_argument("--interval", type=float, metavar="SECONDS",
                    help="batch mode: split every track into fixed-length segments of this "
                         "many seconds (the last short segment is kept). No prompts.")
    args = ap.parse_args()

    if args.stamp and args.interval is not None:
        ap.error("--stamp and --interval are mutually exclusive")
    if args.interval is not None and args.interval <= 0:
        ap.error("--interval must be positive")
    if args.stamp and not os.path.isfile(args.stamp):
        ap.error("stamp file not found: {}".format(args.stamp))
    if not os.path.isdir(args.long):
        ap.error("source folder not found: {}".format(args.long))

    tracks = sorted(
        os.path.join(args.long, f) for f in os.listdir(args.long)
        if f.lower().endswith(AUDIO_EXTS)
    )
    if not tracks:
        print("No audio files found in {}".format(args.long))
        return

    os.makedirs(args.out, exist_ok=True)
    print("Source : {}".format(args.long))
    print("Output : {}".format(args.out))
    print("Tracks : {}".format(len(tracks)))

    if args.stamp:
        print("Mode   : stamp file ({})\n".format(args.stamp))
        done = run_stamp_mode(tracks, args.out, args.stamp)
    elif args.interval is not None:
        print("Mode   : fixed interval ({}s)\n".format(args.interval))
        done = run_interval_mode(tracks, args.out, args.interval)
    else:
        done = 0
        for i, path in enumerate(tracks, 1):
            print("\n[{} / {}]".format(i, len(tracks)))
            result = process_track(path, args.out, play=not args.no_play)
            if result == "quit":
                print("\nStopped by user.")
                break
            if result == "done":
                done += 1

    print("\nFinished. Segmented {} track(s). Output in: {}".format(done, args.out))


if __name__ == "__main__":
    main()
