"""Input source resolution for the vision pipeline.

Turns a user-supplied source string (webcam index, local file, RTSP/HTTP
stream, or YouTube URL) into something OpenCV's ``VideoCapture`` can open.
Kept free of heavy/vision imports so the classification logic is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass

YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
_STREAM_SCHEMES = ("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")


@dataclass
class ResolvedSource:
    kind: str  # "webcam" | "file" | "stream" | "youtube"
    target: object  # int (webcam index) or str (path / URL) for VideoCapture
    label: str  # short human-readable description for the HUD
    is_live: bool  # live feed (always-latest) vs. seekable clip (every frame)


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


def resolve_youtube_url(url: str, fmt: str, timeout_sec: float = 20.0) -> tuple[str, str]:
    """Resolve a YouTube watch URL to a direct media URL via yt-dlp.

    Returns ``(direct_url, title)``. Raises RuntimeError with an actionable
    message when yt-dlp is missing or resolution fails.
    """
    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "YouTube input needs yt-dlp. Install it: pip install yt-dlp"
        ) from exc

    options = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "format": fmt,
        "socket_timeout": timeout_sec,
        "noplaylist": True,
    }
    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not resolve YouTube source: {exc}") from exc

    if info is None:
        raise RuntimeError("Could not resolve YouTube source (no media found).")

    direct = info.get("url")
    if not direct:
        # Some extractions only expose per-format URLs.
        formats = info.get("formats") or []
        for candidate in reversed(formats):
            if candidate.get("url"):
                direct = candidate["url"]
                break
    if not direct:
        raise RuntimeError("YouTube source did not yield a playable stream URL.")

    title = str(info.get("title") or "YouTube") if isinstance(info, dict) else "YouTube"
    return str(direct), title


def resolve_source(
    raw: str | None,
    *,
    camera_index: int,
    youtube_format: str,
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
        direct, title = resolve_youtube_url(value, youtube_format)
        return ResolvedSource(
            kind="youtube",
            target=direct,
            label=_shorten(title),
            is_live=False,
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
