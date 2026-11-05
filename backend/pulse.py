"""Experimental camera pulse estimate (remote photoplethysmography).

Blood volume changes tint facial skin imperceptibly with each heartbeat. The POS
method (Wang et al., IEEE TBME 2017) projects mean skin RGB onto a plane
orthogonal to skin tone, cancelling most lighting and motion, then the dominant
frequency in the 0.7–3 Hz band gives beats per minute. Accuracy depends on
lighting, compression, and stillness, so every estimate carries a confidence
and the UI hides low-confidence readings.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# MediaPipe face-mesh points framing skin patches with few expression wrinkles.
_FOREHEAD_POINTS = (109, 10, 338, 151, 9)
_CHEEK_CENTERS = (205, 425)
_MIN_BPM = 42.0
_MAX_BPM = 180.0


@dataclass(frozen=True)
class PulseReading:
    bpm: float
    confidence: float


def skin_rgb(frame_bgr: np.ndarray, points_px: np.ndarray) -> np.ndarray | None:
    """Mean RGB over forehead and both cheek patches (None when patches are degenerate)."""
    height, width = frame_bgr.shape[:2]
    if len(points_px) <= max(max(_FOREHEAD_POINTS), max(_CHEEK_CENTERS)):
        return None
    face_w = float(points_px[:, 0].max() - points_px[:, 0].min())
    if face_w < 60:
        return None
    patches: list[np.ndarray] = []

    fx = points_px[list(_FOREHEAD_POINTS), 0]
    fy = points_px[list(_FOREHEAD_POINTS), 1]
    x1, x2 = int(fx.min() + face_w * 0.06), int(fx.max() - face_w * 0.06)
    y1, y2 = int(fy.min()), int(fy.max())
    if x2 - x1 > 4 and y2 - y1 > 4:
        patches.append(frame_bgr[max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)])

    half = max(3, int(face_w * 0.06))
    for idx in _CHEEK_CENTERS:
        cx, cy = int(points_px[idx, 0]), int(points_px[idx, 1])
        patches.append(
            frame_bgr[
                max(0, cy - half) : min(height, cy + half),
                max(0, cx - half) : min(width, cx + half),
            ]
        )

    means = [patch.reshape(-1, 3).mean(axis=0) for patch in patches if patch.size >= 27]
    if not means:
        return None
    bgr = np.mean(means, axis=0)
    return bgr[::-1].astype(np.float64)


def pos_signal(rgb: np.ndarray, fs: float) -> np.ndarray:
    """Plane-orthogonal-to-skin pulse signal with overlap-add over 1.6 s windows."""
    n = len(rgb)
    window = max(8, int(round(1.6 * fs)))
    signal = np.zeros(n)
    if n < window:
        return signal
    for end in range(window, n + 1):
        chunk = rgb[end - window : end]
        mean = chunk.mean(axis=0)
        if np.any(mean <= 1e-6):
            continue
        normalized = chunk / mean
        s1 = normalized[:, 1] - normalized[:, 2]
        s2 = normalized[:, 1] + normalized[:, 2] - 2.0 * normalized[:, 0]
        h = s1 + (s1.std() / (s2.std() + 1e-9)) * s2
        signal[end - window : end] += h - h.mean()
    return signal


def dominant_bpm(signal: np.ndarray, fs: float) -> PulseReading | None:
    n = len(signal)
    if n < fs * 6:
        return None
    detrended = signal - np.convolve(signal, np.ones(int(fs)) / int(fs), mode="same")
    tapered = detrended * np.hanning(n)
    padded = 1 << int(np.ceil(np.log2(n * 4)))
    spectrum = np.abs(np.fft.rfft(tapered, padded)) ** 2
    freqs = np.fft.rfftfreq(padded, 1.0 / fs)
    band = (freqs >= _MIN_BPM / 60.0) & (freqs <= _MAX_BPM / 60.0)
    if not np.any(band):
        return None
    band_power = spectrum[band]
    band_freqs = freqs[band]
    peak = int(np.argmax(band_power))
    peak_hz = float(band_freqs[peak])
    near = np.abs(band_freqs - peak_hz) <= 0.1
    signal_power = float(band_power[near].sum())
    noise_power = float(band_power[~near].sum()) + 1e-12
    snr_db = 10.0 * np.log10(signal_power / noise_power)
    confidence = float(np.clip((snr_db + 4.0) / 8.0, 0.0, 1.0))
    return PulseReading(bpm=round(peak_hz * 60.0, 1), confidence=round(confidence, 2))


class PulseEstimator:
    """Rolling rPPG estimate; feed one skin sample per frame with its timestamp."""

    def __init__(self, window_sec: float = 12.0, update_every_sec: float = 1.0) -> None:
        self._window_sec = window_sec
        self._update_every = update_every_sec
        self._samples: deque[tuple[float, np.ndarray]] = deque()
        self._last_update = 0.0
        self._bpm: float | None = None
        self._reading: PulseReading | None = None

    def reset(self) -> None:
        self._samples.clear()
        self._last_update = 0.0
        self._bpm = None
        self._reading = None

    @property
    def reading(self) -> PulseReading | None:
        return self._reading

    def add(self, t: float, rgb: np.ndarray | None, moving: bool = False) -> PulseReading | None:
        if rgb is None or moving:
            if moving and self._samples:
                self._samples.clear()
            return self._reading
        self._samples.append((t, rgb))
        while self._samples and t - self._samples[0][0] > self._window_sec:
            self._samples.popleft()
        if t - self._last_update < self._update_every:
            return self._reading
        self._last_update = t
        self._reading = self._estimate()
        return self._reading

    def _estimate(self) -> PulseReading | None:
        if len(self._samples) < 48:
            return None
        times = np.array([s[0] for s in self._samples])
        span = times[-1] - times[0]
        if span < 6.0:
            return None
        fs = min(30.0, (len(times) - 1) / span)
        if fs < 8.0:
            return None
        grid = np.arange(times[0], times[-1], 1.0 / fs)
        values = np.array([s[1] for s in self._samples])
        rgb = np.stack([np.interp(grid, times, values[:, c]) for c in range(3)], axis=1)
        reading = dominant_bpm(pos_signal(rgb, fs), fs)
        if reading is None:
            return None
        if self._bpm is None or reading.confidence >= 0.5:
            self._bpm = (
                reading.bpm if self._bpm is None else 0.7 * self._bpm + 0.3 * reading.bpm
            )
        return PulseReading(bpm=round(self._bpm, 1), confidence=reading.confidence)
