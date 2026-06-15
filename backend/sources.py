"""Input source resolution for the vision pipeline.

Turns a user-supplied source string (webcam index, local file, RTSP/HTTP
stream, or YouTube URL) into something OpenCV's ``VideoCapture`` can open.
YouTube URLs are downloaded to a temp clip first — direct googlevideo URLs
are not reliably readable by OpenCV on most builds.
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
_STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")
# Android/web clients avoid YouTube SABR/403 issues when fetching media bytes.
YOUTUBE_EXTRACTOR_ARGS = {"youtube": {"player_client": ["android", "web"]}}


@dataclass
class ResolvedSource:
    kind: str  # "webcam" | "file" | "stream" | "youtube"
    target: object  # int (webcam index) or str (path / URL) for VideoCapture
    label: str  # short human-readable description for the HUD
    is_live: bool  # live feed (always-latest) vs. seekable clip (every frame)
    cleanup_path: str | None = None  # temp file to delete when the session ends


def _looks_like_youtube(value: str) -> bool:
    lowered = value.lower()
    return any(host in lowered for host in YOUTUBE_HOSTS)


def _looks_like_stream(value: str) -> bool:
    return value.lower().startswith(_STREAM_SCHEMES)


def classify_source(raw: str | None) -> str:
    """Return the source kind without performing any network/disk resolution."""
    value = (raw or "").strip()
    if not value:
        return "webcam"
    if value.isdigit():
        return "webcam"
    if _looks_like_youtube(value):
        return "youtube"
    if _looks_like_stream(value):
        return "stream"
    return "file"


def _shorten(value: str, limit: int = 48) -> str:
    return value if len(value) <= limit else value[: limit - 1] + "…"


def get_youtube_cache_dir(custom_dir: str | None = None) -> Path:
    """Return the folder where YouTube clips are stored between sessions."""
    if custom_dir:
        path = Path(custom_dir).expanduser().resolve()
    else:
        path = Path(tempfile.gettempdir()) / "yubi-supervision" / "youtube"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prune_youtube_cache(cache_dir: Path, max_age_hours: float) -> int:
    """Remove cached clips older than ``max_age_hours``. Returns files deleted."""
    if max_age_hours <= 0 or not cache_dir.is_dir():
        return 0
    cutoff = time.time() - max_age_hours * 3600.0
    removed = 0
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed


def download_youtube_clip(
    url: str,
    fmt: str,
    timeout_sec: float = 120.0,
    *,
    cache_dir: Path | None = None,
    cache_max_age_hours: float = 168.0,
) -> tuple[str, str]:
    """Download a YouTube watch URL to a local clip OpenCV can read.

    Returns ``(local_path, title)``. Raises RuntimeError with an actionable
    message when yt-dlp is missing or the download fails.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "YouTube input needs yt-dlp. Install it: pip install yt-dlp"
        ) from exc

    out_dir = cache_dir or get_youtube_cache_dir()
    prune_youtube_cache(out_dir, cache_max_age_hours)
    options = {
        "quiet": True,
        "no_warnings": True,
        "format": fmt,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "socket_timeout": timeout_sec,
        "overwrites": True,
        "extractor_args": YOUTUBE_EXTRACTOR_ARGS,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
            if info is None:
                raise RuntimeError("Could not download YouTube video (no media found).")
            path = ydl.prepare_filename(info)
            title = str(info.get("title") or "YouTube")
    except RuntimeError:
        raise
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not download YouTube video: {exc}") from exc

    clip = Path(path)
    if not clip.is_file():
        video_id = str(info.get("id") or "clip")
        ext = str(info.get("ext") or "mp4")
        clip = out_dir / f"{video_id}.{ext}"
    if not clip.is_file():
        raise RuntimeError(
            "YouTube download finished but the clip file is missing on disk."
        )
    return str(clip), title


def resolve_source(
    raw: str | None,
    *,
    camera_index: int,
    youtube_format: str,
    youtube_cache_dir: str | None = None,
    youtube_cache_max_age_hours: float = 168.0,
) -> ResolvedSource:
    """Resolve a source string into a ``ResolvedSource`` ready for VideoCapture.

    YouTube URLs are resolved here (blocking network call) — invoke from a
    worker thread, never the event loop.
    """
    value = (raw or "").strip()
    kind = classify_source(value)

    if kind == "webcam":
        index = int(value) if value.isdigit() else camera_index
        return ResolvedSource(
            kind="webcam",
            target=index,
            label=f"Webcam {index}",
            is_live=True,
        )

    if kind == "youtube":
        cache_dir = get_youtube_cache_dir(youtube_cache_dir)
        clip_path, title = download_youtube_clip(
            value,
            youtube_format,
            cache_dir=cache_dir,
            cache_max_age_hours=youtube_cache_max_age_hours,
        )
        return ResolvedSource(
            kind="youtube",
            target=clip_path,
            label=_shorten(title),
            is_live=False,
            cleanup_path=clip_path,
        )

    if kind == "stream":
        return ResolvedSource(
            kind="stream",
            target=value,
            label=_shorten(value),
            is_live=True,
        )

    # Local file / clip.
    name = value.rsplit("/", 1)[-1] or value
    return ResolvedSource(
        kind="file",
        target=value,
        label=_shorten(name),
        is_live=False,
    )
