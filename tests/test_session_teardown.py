from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from backend.live_capture import LatestFrameSlot, OpenCvReader
from backend.sources import ResolvedSource
from backend.vision import VisionEngine, _CaptureWorker


def test_open_video_capture_uses_default_backend_for_files() -> None:
    engine = VisionEngine()
    resolved = ResolvedSource(
        kind="file",
        target="/tmp/test.mp4",
        label="test.mp4",
        is_live=False,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch("backend.vision.cv2.VideoCapture") as mock_ctor:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        mock_ctor.return_value = cap
        engine._open_video_capture(resolved)
    mock_ctor.assert_called_once_with("/tmp/test.mp4")


def test_open_video_capture_uses_ffmpeg_for_live_streams() -> None:
    engine = VisionEngine()
    resolved = ResolvedSource(
        kind="stream",
        target="rtsp://10.0.0.5/stream",
        label="cam",
        is_live=True,
    )
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    with patch("backend.vision.cv2.VideoCapture") as mock_ctor:
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, frame)
        mock_ctor.return_value = cap
        engine._open_video_capture(resolved)
    mock_ctor.assert_called_once_with("rtsp://10.0.0.5/stream", cv2.CAP_FFMPEG)


def test_end_session_from_loop_stops_capture_worker() -> None:
    cap = MagicMock()
    cap.read.return_value = (False, None)
    reader = OpenCvReader(cap)
    slot = LatestFrameSlot()
    worker = _CaptureWorker(reader, slot)
    worker.start()
    time.sleep(0.05)

    engine = VisionEngine()
    engine._capture_worker = worker
    engine._frame_reader = reader
    engine._running = True
    engine._end_session_from_loop()

    assert engine._capture_worker is None
    assert engine.get_stats().state == "idle"
    assert engine.is_running is False
