from __future__ import annotations

from backend.vision import VisionEngine


def test_remote_live_stream_only_when_source_is_live() -> None:
    engine = VisionEngine()

    engine._source_kind = "youtube"
    engine._source_is_live = False
    assert engine._is_remote_live_stream() is False

    engine._source_is_live = True
    assert engine._is_remote_live_stream() is True

    engine._source_kind = "file"
    engine._source_is_live = False
    assert engine._is_remote_live_stream() is False

    engine._source_kind = "stream"
    engine._source_is_live = True
    assert engine._is_remote_live_stream() is True
