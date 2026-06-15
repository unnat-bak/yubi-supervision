from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
from backend.live_capture import LatestFrameSlot, OpenCvReader
from backend.vision import _CaptureWorker


def test_capture_worker_loops_seekable_source_on_eof() -> None:
    cap = MagicMock()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    cap.read.side_effect = [(True, frame), (False, None), (True, frame)]
    reader = OpenCvReader(cap)
    slot = LatestFrameSlot()
    worker = _CaptureWorker(reader, slot, drop_frames=False, loop_on_eof=True)
    worker.start()

    try:
        deadline = time.time() + 2.0
        while time.time() < deadline:
            ok, got = worker.read_latest(timeout=0.2)
            if ok and got is not None:
                break
            time.sleep(0.01)
        assert ok and got is not None
        assert cap.set.called
        assert worker.ended is False
    finally:
        worker.stop()
