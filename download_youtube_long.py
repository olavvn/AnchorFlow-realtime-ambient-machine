"""Download long ambient tracks from a YouTube playlist into the `long` folder.

This is the first step of the segmentation workflow:
    1. (this script)        download tracks        -> <dest>/NNN-<title>.wav
    2. segment_audio.py     listen + cut by hand   -> <out>/<title>/<title>-NNN.wav

Every downloaded file gets a zero-padded numeric prefix (001-, 002-, ...) so it
keeps the playlist order and lines up with a --stamp file in segment_audio.py.
Numbering continues from the highest prefix already in the folder, and tracks
already in the download archive are skipped (they consume no number).

Videos that HAVE chapters are split into one file per chapter, and each chapter
takes the next number in sequence. E.g. if 006 is next and a video has 3
chapters, they become 006-, 007-, 008-.

Requires `yt-dlp` and `ffmpeg` on PATH (both already installed on this machine).

Usage:
    python download_youtube_long.py <PLAYLIST_URL> [<URL> ...]
    python download_youtube_long.py -f urls.txt
    python download_youtube_long.py <URL> --audio-format flac --cookies-from-browser chrome
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

YTDLP = "yt-dlp"
DEFAULT_DEST = r"C:\Users\Changhyun Kim\Desktop\SG_Lecture\2026-01\DL for Music & Audio\datasets\ambient_youtube\long"


def highest_index(dest):
    """Largest NNN- prefix already present in dest, so new files continue from there."""
    if not os.path.isdir(dest):
        return 0
    nums = [int(m.group(1)) for f in os.listdir(dest)
            if (m := re.match(r"(\d+)-", f))]
    return max(nums, default=0)


def run(cmd):
    print(">>", " ".join(cmd))
    return subprocess.run(cmd).returncode


def expand(url, extra):
    """Expand a playlist URL into individual video URLs; pass a single video through."""
    res = subprocess.run([YTDLP, "--flat-playlist", "-J", *extra, url], capture_output=True, text=True)
    if res.returncode != 0:
        print("  [list failed] {}".format(url), file=sys.stderr)
        print(res.stderr.strip()[-500:], file=sys.stderr)
        return []
    data = json.loads(res.stdout)
    if data.get("_type") == "playlist":
        out = [e.get("url") or e.get("webpage_url") or e.get("id")
               for e in (data.get("entries") or []) if e]
        out = [u for u in out if u]
        print("[playlist: {} item(s)] {}".format(len(out), data.get("title", "playlist")))
        return out
    return [data.get("webpage_url") or data.get("original_url") or url]


def download_video(url, idx, dest, audio_format, archive, extra):
    # Prefix with a zero-padded index so files keep the playlist order
    # (e.g. 001-title.wav) and line N of a --stamp file maps to track N.
    full_out = os.path.join(dest, "{:03d}-%(title)s.%(ext)s".format(idx))
    run([
        YTDLP, "--no-playlist", *extra,
        "-x", "--audio-format", audio_format,
        "--restrict-filenames",
        "--download-archive", archive,
        "-o", full_out,
        url,
    ])


def main():
    ap = argparse.ArgumentParser(
        description="Download full YouTube tracks (no splitting) into the `long` folder.")
    ap.add_argument("urls", nargs="*", help="one or more YouTube URLs (playlist or single video)")
    ap.add_argument("-f", "--urls-file", help="text file with one URL per line (# = comment)")
    ap.add_argument("-d", "--dest", default=DEFAULT_DEST, help="destination folder")
    ap.add_argument("--audio-format", default="wav", help="audio format (wav, flac, mp3, ...). default: wav")
    ap.add_argument("--cookies-from-browser",
                    help="read cookies from this browser (chrome, edge, firefox, ...) to bypass YouTube bot check")
    ap.add_argument("--cookies", help="path to a cookies.txt file (alternative to --cookies-from-browser)")
    args = ap.parse_args()

    extra = []
    if args.cookies_from_browser:
        extra += ["--cookies-from-browser", args.cookies_from_browser]
    if args.cookies:
        extra += ["--cookies", args.cookies]

    urls = list(args.urls)
    if args.urls_file:
        with open(args.urls_file, encoding="utf-8") as fh:
            urls += [ln.strip() for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
    if not urls:
        ap.error("no URLs given (pass URLs as arguments or via -f urls.txt)")

    os.makedirs(args.dest, exist_ok=True)
    archive = os.path.join(args.dest, "downloaded_archive.txt")
    start = highest_index(args.dest)
    print("Destination : {}".format(args.dest))
    print("Audio format: {}".format(args.audio_format))
    print("Numbering   : continuing from {:03d}\n".format(start + 1))

    n = 0
    for url in urls:
        for video_url in expand(url, extra):
            n += 1
            download_video(video_url, start + n, args.dest, args.audio_format, archive, extra)

    print("\nDone. Processed {} track(s). Files in: {}".format(n, args.dest))


if __name__ == "__main__":
    main()
