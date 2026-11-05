"""Low-latency smoothing for landmark and blendshape streams."""

from __future__ import annotations

import math

import numpy as np


def _smoothing_factor(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """One Euro filter (Casiez et al., CHI 2012) over arrays of any shape.

    A low cutoff while the signal is still removes jitter; the cutoff rises with
    speed, so genuine fast movement (a 200 ms micro-expression, a head turn)
    passes through without the lag a fixed low-pass filter would add.
    """

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None
        self._t: float | None = None

    def reset(self) -> None:
        self._x = None
        self._dx = None
        self._t = None

    def __call__(self, value: np.ndarray, t: float) -> np.ndarray:
        x = np.asarray(value, dtype=np.float32)
        if self._x is None or self._t is None or self._x.shape != x.shape:
            self._x = x.copy()
            self._dx = np.zeros_like(x)
            self._t = t
            return x.copy()
        dt = t - self._t
        if dt <= 1e-6:
            return self._x.copy()
        self._t = t
        dx = (x - self._x) / dt
        self._dx = self._dx + _smoothing_factor(self.d_cutoff, dt) * (dx - self._dx)
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        self._x = self._x + _smoothing_factor(cutoff, dt) * (x - self._x)
        return self._x.copy()


def ema_alpha(dt: float, tau_sec: float) -> float:
    """Frame-rate independent EMA weight for a time constant of ``tau_sec``."""
    if tau_sec <= 0:
        return 1.0
    return 1.0 - math.exp(-max(dt, 0.0) / tau_sec)
