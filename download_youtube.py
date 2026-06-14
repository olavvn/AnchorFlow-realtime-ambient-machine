"""Download ambient tracks from YouTube, splitting by chapters when present.

Each video is saved as audio (wav by default) into the destination folder.
- If the video HAS chapters  -> one file per chapter under  <dest>/<title>/NN-<chapter>.wav
- If the video has NO chapters -> a single file           <dest>/<title>.wav
- If the URL is a PLAYLIST    -> every video in it is downloaded individually (per above)

Requires `yt-dlp` and `ffmpeg` on PATH (both already installed on this machine).

Usage:
    python download_youtube.py <URL> [<URL> ...]
    python download_youtube.py -f urls.txt
    python download_youtube.py <URL> --audio-format flac --dest "D:/some/folder"
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

YTDLP = "yt-dlp"
DEFAULT_DEST = r"C:\Users\Changhyun Kim\Desktop\SG_Lecture\2026-01\DL for Music & Audio\datasets\ambient_youtube"


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


def probe(url, extra):
    """Full extraction of a single video -> info dict (or None on failure)."""
    res = subprocess.run([YTDLP, "--no-playlist", "-J", *extra, url], capture_output=True, text=True)
    if res.returncode != 0:
        print("  [probe failed] {}".format(url), file=sys.stderr)
        print(res.stderr.strip()[-500:], file=sys.stderr)
        return None
    return json.loads(res.stdout)


def download_video(info, dest, audio_format, archive, extra):
    url = info.get("webpage_url") or info.get("original_url") or info.get("id")
    title = info.get("title", "video")
    chapters = info.get("chapters") or []

    common = [
        YTDLP, "--no-playlist", *extra,
        "-x", "--audio-format", audio_format,
        "--restrict-filenames",
        "--download-archive", archive,
    ]

    if chapters:
        print("[chapters: {}] {}".format(len(chapters), title))
        chapter_out = os.path.join(dest, "%(title)s", "%(section_number)02d-%(section_title)s.%(ext)s")
        # The non-split full file is redundant when chapters exist -> send it to a
        # temp dir that is auto-deleted, keeping only the per-chapter files.
        with tempfile.TemporaryDirectory() as tmp:
            full_out = os.path.join(tmp, "%(title)s.%(ext)s")
            run(common + [
                "--split-chapters",
                "-o", full_out,
                "-o", "chapter:" + chapter_out,
                url,
            ])
    else:
        print("[no chapters] {}".format(title))
        full_out = os.path.join(dest, "%(title)s.%(ext)s")
        run(common + ["-o", full_out, url])


def main():
    ap = argparse.ArgumentParser(description="Download YouTube audio, split by chapters when present.")
    ap.add_argument("urls", nargs="*", help="one or more YouTube URLs")
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
    print("Destination : {}".format(args.dest))
    print("Audio format: {}\n".format(args.audio_format))

    n_videos = 0
    for url in urls:
        for video_url in expand(url, extra):
            info = probe(video_url, extra)
            if info is None:
                continue
            download_video(info, args.dest, args.audio_format, archive, extra)
            n_videos += 1

    print("\nDone. Processed {} video(s). Files in: {}".format(n_videos, args.dest))


if __name__ == "__main__":
    main()
