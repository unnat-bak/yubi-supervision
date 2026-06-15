"""Low-latency frame sources for remote live feeds (YouTube HLS, RTSP via ffmpeg pipe)."""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Any, Protocol

import numpy as np


class FrameReader(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...

    def release(self) -> None: ...


class LatestFrameSlot:
    """Thread-safe holder for the newest BGR frame and a monotonic sequence id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._frame: np.ndarray | None = None

    def publish(self, frame: np.ndarray) -> int:
        with self._lock:
            self._seq += 1
            self._frame = frame
            return self._seq

    def snapshot(self) -> tuple[int, np.ndarray | None]:
        """Return ``(seq, frame_copy)`` for consumers that need an owned buffer."""
        with self._lock:
            if self._frame is None:
                return self._seq, None
            return self._seq, self._frame.copy()

    def peek(self) -> tuple[int, np.ndarray | None]:
        """Return ``(seq, frame_ref)`` without copying — valid until next publish."""
        with self._lock:
            return self._seq, self._frame

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq


def fit_stream_dimensions(
    width: int | None,
    height: int | None,
    *,
    max_height: int = 720,
) -> tuple[int, int]:
    w = max(2, int(width or 1280))
    h = max(2, int(height or 720))
    if h > max_height:
        scale = max_height / h
        w = max(2, int(w * scale))
        h = max_height
    # ffmpeg scale prefers even dimensions
    w -= w % 2
    h -= h % 2
    return w, h


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


class FfmpegPipeReader:
    """Read raw BGR frames from an ffmpeg subprocess (low-latency HLS/RTSP)."""

    def __init__(self, url: str, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._frame_size = width * height * 3
        self._ended = False
        self._released = False
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-user_agent",
                "Mozilla/5.0",
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
                "-fflags",
                "nobuffer+discardcorrupt",
                "-flags",
                "low_delay",
                "-i",
                url,
                "-an",
                "-vf",
                f"scale={width}:{height}",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if self._proc.stdout is None:
            self.release()
            raise RuntimeError("ffmpeg pipe stdout unavailable")

    @property
    def ended(self) -> bool:
        with self._lock:
            if self._released or self._ended:
                return True
            proc = self._proc
        return proc is None or proc.poll() is not None

    def interrupt(self) -> None:
        """Unblock a capture thread stuck in ``read()`` (safe from any thread)."""
        with self._lock:
            if self._released:
                return
            self._ended = True
            proc = self._proc
            stdout = proc.stdout if proc is not None else None
        if stdout is not None:
            try:
                stdout.close()
            except Exception:
                pass
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._released or self._ended:
                return False, None
            proc = self._proc
            stdout = proc.stdout if proc is not None else None
        if proc is None or stdout is None:
            return False, None
        buf = bytearray()
        deadline = time.time() + 5.0
        while len(buf) < self._frame_size:
            if proc.poll() is not None:
                with self._lock:
                    self._ended = True
                return False, None
            if time.time() > deadline:
                return False, None
            try:
                chunk = stdout.read(self._frame_size - len(buf))
            except (ValueError, OSError):
                with self._lock:
                    self._ended = True
                return False, None
            if chunk:
                buf.extend(chunk)
            else:
                time.sleep(0.01)
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(
            (self._height, self._width, 3)
        )
        return True, frame.copy()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            self._ended = True
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        stdout = proc.stdout
        if stdout is not None:
            try:
                stdout.close()
            except Exception:
                pass
        try:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=2.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass


class OpenCvReader:
    """Thin wrapper so ``VideoCapture`` matches ``FrameReader``."""

    def __init__(self, cap: Any) -> None:
        self._cap = cap
        self._ended = False
        self._released = False

    @property
    def ended(self) -> bool:
        return self._ended or self._released

    def interrupt(self) -> None:
        """Mark ended; ``VideoCapture.release`` must run on the capture thread."""
        self._ended = True

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._released or self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if not ok:
            self._ended = True
            return False, None
        self._ended = False
        return True, frame.copy()

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        cap = self._cap
        self._cap = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass


def open_ffmpeg_pipe(url: str, width: int, height: int) -> FfmpegPipeReader:
    if not ffmpeg_available():
        raise RuntimeError("ffmpeg not found on PATH")
    deadline = time.time() + 15.0
    last_error = "could not read first frame"
    while time.time() < deadline:
        reader = FfmpegPipeReader(url, width, height)
        ok, _ = reader.read()
        if ok:
            return reader
        last_error = "ffmpeg pipe opened but could not read first frame"
        reader.release()
        time.sleep(0.4)
    raise RuntimeError(last_error)
