from __future__ import annotations

import numpy as np
import pytest
from backend.facs import (
    AUEvent,
    contempt_score,
    dominant_emotion,
    episode_cues,
    read_action_units,
    score_emotions,
)
from backend.pulse import PulseEstimator
from backend.signal_filters import OneEuroFilter, ema_alpha


def test_one_euro_suppresses_jitter_but_tracks_steps() -> None:
    rng = np.random.default_rng(1)
    filt = OneEuroFilter(min_cutoff=1.2, beta=30.0)
    out = [filt(np.array([0.5 + rng.normal(0, 0.002)]), i / 30.0)[0] for i in range(60)]
    assert np.std(out[10:]) < 0.002 * 0.6
    step = [filt(np.array([0.7]), (60 + i) / 30.0)[0] for i in range(6)]
    assert step[-1] > 0.68


def test_ema_alpha_is_frame_rate_independent() -> None:
    per_frame_30 = 1 - (1 - ema_alpha(1 / 30, 2.0)) ** 30
    per_frame_60 = 1 - (1 - ema_alpha(1 / 60, 2.0)) ** 60
    assert per_frame_30 == pytest.approx(per_frame_60)
    assert ema_alpha(0.1, 0.0) == 1.0


def test_pulse_estimator_recovers_synthetic_heart_rate() -> None:
    rng = np.random.default_rng(5)
    estimator = PulseEstimator(window_sec=12.0)
    reading = None
    for i in range(int(14 * 30)):
        t = i / 30.0
        wave = 0.6 * np.sin(2 * np.pi * 1.2 * t)
        rgb = np.array([150 + 0.3 * wave, 110 + wave, 90 + 0.2 * wave]) + rng.normal(0, 0.05, 3)
        reading = estimator.add(t, rgb)
    assert reading is not None
    assert reading.bpm == pytest.approx(72.0, abs=4.0)
    assert reading.confidence > 0.3


def test_pulse_estimator_drops_window_on_motion() -> None:
    estimator = PulseEstimator()
    for i in range(100):
        estimator.add(i / 30.0, np.array([120.0, 100.0, 90.0]))
    estimator.add(3.4, None, moving=True)
    assert estimator._samples == type(estimator._samples)()


def test_duchenne_smile_scores_joy() -> None:
    act = {"AU6": 0.8, "AU12": 0.9}
    scores = score_emotions(act, act, act)
    assert dominant_emotion(scores)[0] == "happiness"


def test_contempt_requires_asymmetry() -> None:
    assert contempt_score({"AU12": 0.7, "AU14": 0.7}, {"AU12": 0.0, "AU14": 0.0}) > 0.3
    assert contempt_score({"AU12": 0.7, "AU14": 0.7}, {"AU12": 0.7, "AU14": 0.7}) < 0.1


def test_lateral_action_units_keep_sides() -> None:
    readings = read_action_units({"mouthSmileLeft": 0.6, "mouthSmileRight": 0.1})
    assert readings["AU12"].value == pytest.approx(0.6)
    assert readings["AU12"].left > readings["AU12"].right


def test_masking_and_suppression_cues() -> None:
    negative = AUEvent(code="AU15", onset=10.0, offset=10.3, apex_t=10.1, apex_level=0.6, side="")
    masking = episode_cues([negative], onset=10.0, duration=0.3, smile_onset=10.35)
    assert "masking" in masking
    pressed = episode_cues([negative], onset=10.0, duration=0.3, press_onset=10.2)
    assert "suppression" in pressed
    leaked = episode_cues([negative], onset=10.0, duration=0.3, smile_onset=8.5)
    assert "leakage" in leaked
