from __future__ import annotations

from unittest.mock import patch

from backend.sources import classify_source, resolve_source


def test_classify_webcam() -> None:
    assert classify_source(None) == "webcam"
    assert classify_source("") == "webcam"
    assert classify_source("  ") == "webcam"
    assert classify_source("0") == "webcam"
    assert classify_source("2") == "webcam"


def test_classify_youtube() -> None:
    assert classify_source("https://www.youtube.com/watch?v=abc123") == "youtube"
    assert classify_source("https://youtu.be/abc123") == "youtube"
    assert classify_source("https://www.youtube.com/live/EO_1LWqsCNE") == "youtube"


def test_classify_youtube_live_url_not_stream() -> None:
    """https:// live page URLs must route to YouTube, not generic HTTP stream."""
    url = "https://www.youtube.com/live/EO_1LWqsCNE"
    assert classify_source(url) == "youtube"
    assert classify_source(url) != "stream"


def test_classify_stream_and_file() -> None:
    assert classify_source("rtsp://10.0.0.5:554/stream") == "stream"
    assert classify_source("http://cam.local/feed.mjpg") == "stream"
    assert classify_source("clips/traffic.mp4") == "file"
    assert classify_source("/abs/path/scene.mov") == "file"


def test_resolve_webcam_uses_index() -> None:
    resolved = resolve_source("", camera_index=1, youtube_format="best")
    assert resolved.kind == "webcam"
    assert resolved.target == 1
    assert resolved.is_live is True

    explicit = resolve_source("3", camera_index=1, youtube_format="best")
    assert explicit.target == 3


def test_resolve_file_is_not_live() -> None:
    resolved = resolve_source(
        "clips/match.mp4", camera_index=0, youtube_format="best"
    )
    assert resolved.kind == "file"
    assert resolved.target == "clips/match.mp4"
    assert resolved.is_live is False
    assert resolved.label == "match.mp4"


def test_resolve_stream_is_live() -> None:
    resolved = resolve_source(
        "rtsp://10.0.0.5:554/stream", camera_index=0, youtube_format="best"
    )
    assert resolved.kind == "stream"
    assert resolved.is_live is True


def test_resolve_youtube_downloads_local_clip() -> None:
    vod_info = {"title": "CCTV clip", "is_live": False, "live_status": "not_live"}
    with patch("backend.sources.probe_youtube", return_value=vod_info), patch(
        "backend.sources.download_youtube_clip",
        return_value=("/tmp/yubi-test.mp4", "CCTV clip"),
    ) as download:
        resolved = resolve_source(
            "https://youtu.be/abc123XYZ01",
            camera_index=0,
            youtube_format="best[ext=mp4]/best",
        )
    download.assert_called_once()
    assert download.call_args.args[0] == "https://youtu.be/abc123XYZ01"
    assert download.call_args.args[1] == "best[ext=mp4]/best"
    assert resolved.kind == "youtube"
    assert resolved.target == "/tmp/yubi-test.mp4"
    assert resolved.cleanup_path == "/tmp/yubi-test.mp4"
    assert resolved.is_live is False
    assert resolved.label == "CCTV clip"


def test_resolve_youtube_live_stream() -> None:
    live_info = {
        "title": "City cam",
        "is_live": True,
        "live_status": "is_live",
        "url": "https://cdn.example/live.m3u8",
    }
    with patch("backend.sources.probe_youtube", return_value=live_info):
        resolved = resolve_source(
            "https://www.youtube.com/live/EO_1LWqsCNE",
            camera_index=0,
            youtube_format="best[ext=mp4]/best",
        )
    assert resolved.kind == "youtube"
    assert resolved.target == "https://cdn.example/live.m3u8"
    assert resolved.is_live is True
    assert resolved.cleanup_path is None
    assert "LIVE" in resolved.label


def test_refresh_youtube_live_stream() -> None:
    live_info = {
        "is_live": True,
        "live_status": "is_live",
        "url": "https://cdn.example/refreshed.m3u8",
    }
    with patch("backend.sources.probe_youtube", return_value=live_info):
        from backend.sources import refresh_youtube_live_stream

        url = refresh_youtube_live_stream("https://www.youtube.com/live/EO_1LWqsCNE")
    assert url == "https://cdn.example/refreshed.m3u8"


def test_resolve_youtube_upcoming_raises() -> None:
    with patch(
        "backend.sources.probe_youtube",
        return_value={"live_status": "is_upcoming", "title": "Soon"},
    ):
        try:
            resolve_source(
                "https://youtu.be/upcoming",
                camera_index=0,
                youtube_format="best",
            )
            raised = False
        except RuntimeError as exc:
            raised = True
            assert "not started" in str(exc).lower()
    assert raised


def test_prune_youtube_cache_removes_old_files(tmp_path) -> None:
    import os
    import time

    from backend.sources import prune_youtube_cache

    old = tmp_path / "old.mp4"
    new = tmp_path / "new.mp4"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    old_time = time.time() - 3600 * 200
    os.utime(old, (old_time, old_time))

    removed = prune_youtube_cache(tmp_path, max_age_hours=168)
    assert removed == 1
    assert not old.exists()
    assert new.exists()
