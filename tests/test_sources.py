from __future__ import annotations

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
