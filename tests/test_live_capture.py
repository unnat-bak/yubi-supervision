"""Unit tests for low-latency live capture helpers."""

from __future__ import annotations

import numpy as np
from backend.live_capture import (
    LatestFrameSlot,
    OpenCvReader,
    ffmpeg_available,
    fit_stream_dimensions,
)


def test_latest_frame_slot_seq_increments() -> None:
    slot = LatestFrameSlot()
    frame_a = np.zeros((4, 4, 3), dtype=np.uint8)
    frame_b = np.ones((4, 4, 3), dtype=np.uint8)

    seq_a = slot.publish(frame_a)
    seq_b = slot.publish(frame_b)

    assert seq_a == 1
    assert seq_b == 2
    assert slot.seq == 2

    peek_seq, peek_frame = slot.peek()
    assert peek_seq == 2
    assert peek_frame is frame_b

    snap_seq, snap_frame = slot.snapshot()
    assert snap_seq == 2
    assert snap_frame is not None
    assert snap_frame.shape == (4, 4, 3)
    snap_frame[0, 0, 0] = 99
    assert peek_frame[0, 0, 0] != 99


def test_fit_stream_dimensions_caps_height_and_even_sizes() -> None:
    w, h = fit_stream_dimensions(1920, 1080, max_height=720)
    assert h == 720
    assert w % 2 == 0
    assert h % 2 == 0


def test_ffmpeg_available_is_bool() -> None:
    assert isinstance(ffmpeg_available(), bool)


def test_opencv_reader_release_is_idempotent() -> None:
    from unittest.mock import MagicMock

    cap = MagicMock()
    cap.read.return_value = (True, np.zeros((2, 2, 3), dtype=np.uint8))
    reader = OpenCvReader(cap)
    reader.release()
    reader.release()
    cap.release.assert_called_once()
    ok, frame = reader.read()
    assert ok is False and frame is None


def test_opencv_reader_seek_to_start_rewinds_capture() -> None:
    from unittest.mock import MagicMock

    cap = MagicMock()
    reader = OpenCvReader(cap)
    reader._ended = True

    assert reader.seek_to_start() is True
    cap.set.assert_called_once()
    assert reader.ended is False
