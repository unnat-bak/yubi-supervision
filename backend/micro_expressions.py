"""Per-subject micro-expression engine — runs locally on every frame, no API cost.

Pipeline per tracked face:

1. Blendshapes -> FACS action units (``backend.facs``), lightly smoothed with a
   One Euro filter so noise drops without blunting 100–500 ms movements.
2. Each AU keeps an adaptive baseline (the subject's resting face) and noise
   spread. An onset fires when the AU rises well above both; hysteresis tracks
   it through apex to offset.
3. AU events whose onsets cluster within 200 ms form one expression episode,
   classified as a micro-expression (<= 500 ms) or a macro expression, matched
   against EMFACS emotion prototypes, and annotated with cues (Duchenne smile,
   masking, suppression, leakage, …).
4. Parallel detectors track blinks, gaze aversion, head nods/shakes, speech,
   hand-to-face contact, and an experimental camera pulse estimate.

Timestamps are media time for files (so findings map onto the video) and
session time for live sources.
"""

from __future__ import annotations

import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from backend.config import Settings
from backend.facs import (
    ACTION_UNITS,
    AU_BY_CODE,
    CUES,
    EMOTION_LABELS,
    MICRO_MAX_SEC,
    NEGATIVE_EMOTIONS,
    REGIONS,
    AUEvent,
    AUReading,
    activation,
    blendshape_scores,
    describe_aus,
    dominant_emotion,
    emotion_involves,
    episode_cues,
    is_negative_event,
    read_action_units,
    region_levels,
    score_emotions,
    valence,
)
from backend.pulse import PulseEstimator, skin_rgb
from backend.signal_filters import OneEuroFilter, ema_alpha

SUSTAINED_SEC = 8.0
_CLUSTER_WINDOW_SEC = 0.2
_CLUSTER_GRACE_SEC = 0.12
# Negative clusters wait briefly so a following smile / lip press can be attributed.
_NEGATIVE_CLUSTER_GRACE_SEC = 0.6
# Speech detection needs a few syllables; jaw / lip-only clusters wait for it to settle.
_SPEECH_ONSET_GRACE_SEC = 0.8
_Z_ONSET = 3.0
_MIN_SPREAD = 0.012
_HEAD_FAST_DEG_S = 80.0
_SUBJECT_TIMEOUT_SEC = 3.0
_SUBJECT_REACQUIRE_SEC = 12.0
_GAP_RESET_SEC = 1.0
_STATUS_INTERVAL_SEC = 0.2
_LIVE_TIMELINE_POINTS = 120
_LIVE_TIMELINE_STEP_SEC = 0.25
_CROP_INTERVAL_SEC = 0.12
_CROP_BUFFER_SEC = 2.5
_CROP_MAX_WIDTH = 192
_MAX_EPISODES = 5000
_FLASH_SEC = 0.9
_SUBJECT_NAMES = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_EYE_AUS = frozenset({"AU5", "AU6", "AU7"})
_LOWER_FACE_REGIONS = frozenset({"mouth", "jaw", "cheek"})
_BROW_RAISE_AUS = frozenset({"AU1", "AU2"})
_BROW_AUS = frozenset({"AU1", "AU2", "AU4"})
_BATON_MAX_SEC = 1.0
# Articulation moves the mouth constantly; these cues need a quiet lower face.
_SPEECH_CONFOUNDED_CUES = frozenset({"masking", "suppression", "leakage", "lip_compression"})
_MIN_EPISODE_LEVEL = 0.3
_PULSE_SHOW_CONFIDENCE = 0.3
_SPEAKING_LOWER_FACE_MIN_LEVEL = 0.6


@dataclass
class Episode:
    id: str
    subject: str
    kind: str
    label: str
    at: float
    t: float
    duration_ms: int
    intensity: float
    aus: list[str] = field(default_factory=list)
    emotion: str | None = None
    emotion_score: float = 0.0
    side: str = ""
    cues: list[str] = field(default_factory=list)
    speaking: bool = False
    apex_crop: np.ndarray | None = field(default=None, repr=False)

    @property
    def notable(self) -> bool:
        if self.kind == "micro":
            return True
        return any(
            cue in {"masking", "suppression", "leakage", "unilateral_smirk"} for cue in self.cues
        ) or (self.emotion in NEGATIVE_EMOTIONS and self.emotion_score >= 0.35)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject": self.subject,
            "kind": self.kind,
            "label": self.label,
            "at": round(self.at, 2),
            "duration_ms": self.duration_ms,
            "intensity": round(self.intensity, 2),
            "aus": list(self.aus),
            "au_label": describe_aus(self.aus) if self.aus else "",
            "emotion": self.emotion,
            "emotion_score": round(self.emotion_score, 2),
            "side": self.side,
            "cues": list(self.cues),
            "cue_labels": [CUES[c].label for c in self.cues if c in CUES],
            "speaking": self.speaking,
            "notable": self.notable,
        }


@dataclass
class SubjectRender:
    subject: str
    points_px: np.ndarray
    regions: dict[str, float]
    tag: str
    flash: str | None
    flash_strength: float
    primary: bool


@dataclass
class ExpressionFrame:
    renders: list[SubjectRender]
    episodes: list[Episode]


class _AUChannel:
    """Baseline, noise spread, and onset/apex/offset state for one action unit."""

    def __init__(self, code: str) -> None:
        self.au = AU_BY_CODE[code]
        self.baseline = 0.0
        self.baseline_l = 0.0
        self.baseline_r = 0.0
        self.spread = 0.03
        self.active = False
        self.holding = False
        self.pending = 0
        self.pending_since = 0.0
        self.onset = 0.0
        self.apex_t = 0.0
        self.apex_delta = 0.0
        self.apex_l = 0.0
        self.apex_r = 0.0
        self.cluster: int | None = None
        self.seeded = False

    def level(self, value: float) -> float:
        return max(0.0, min(1.0, (value - self.baseline) / self.au.gain))

    def side_level(self, value: float, left: bool) -> float:
        base = self.baseline_l if left else self.baseline_r
        return max(0.0, min(1.0, (value - base) / self.au.gain))

    def cancel(self) -> None:
        self.active = False
        self.holding = False
        self.pending = 0
        self.cluster = None

    def _close(self, t: float) -> AUEvent:
        side = ""
        left = max(0.0, self.apex_l)
        right = max(0.0, self.apex_r)
        if self.au.lateral:
            strong = max(left, right)
            if strong > 0 and min(left, right) < 0.45 * strong:
                side = "L" if left > right else "R"
            else:
                side = "bilateral"
        gain = self.au.gain
        return AUEvent(
            code=self.au.code,
            onset=self.onset,
            offset=t,
            apex_t=self.apex_t,
            apex_level=round(min(1.0, self.apex_delta / gain), 3),
            side=side,
            left=round(min(1.0, left / gain), 3),
            right=round(min(1.0, right / gain), 3),
        )

    def step(
        self,
        reading: AUReading,
        t: float,
        dt: float,
        *,
        calibrating: bool,
        allow_onset: bool,
        threshold_scale: float,
        sensitivity: float,
    ) -> tuple[AUEvent | None, bool]:
        """Advance one frame. Returns (closed event, just_activated)."""
        if not self.seeded:
            self.baseline = reading.value
            self.baseline_l = reading.left
            self.baseline_r = reading.right
            self.seeded = True
        delta = reading.value - self.baseline
        floor = self.au.min_delta * threshold_scale / sensitivity
        need = max(floor, _Z_ONSET * self.spread / sensitivity)
        event: AUEvent | None = None
        activated = False

        if calibrating:
            self.cancel()
        elif self.active or self.holding:
            if self.active and delta > self.apex_delta:
                self.apex_delta = delta
                self.apex_t = t
                self.apex_l = reading.left - self.baseline_l
                self.apex_r = reading.right - self.baseline_r
            release = max(0.35 * self.apex_delta, 0.5 * floor)
            if delta < release:
                if self.active:
                    event = self._close(t)
                self.active = False
                self.holding = False
            elif self.active and t - self.onset >= SUSTAINED_SEC:
                event = self._close(t)
                self.active = False
                self.holding = True
        elif allow_onset and delta > need:
            if self.pending == 0:
                self.pending_since = t
            self.pending += 1
            if self.pending >= 2:
                self.active = True
                activated = True
                self.pending = 0
                self.onset = self.pending_since
                self.apex_delta = delta
                self.apex_t = t
                self.apex_l = reading.left - self.baseline_l
                self.apex_r = reading.right - self.baseline_r
        else:
            self.pending = 0

        if calibrating:
            tau = 0.8
        elif delta < 0:
            tau = 2.5
        elif self.active or self.holding or delta > 2.5 * self.spread:
            tau = 90.0
        else:
            tau = 8.0
        alpha = ema_alpha(dt, tau)
        self.baseline += alpha * (reading.value - self.baseline)
        self.baseline_l += alpha * (reading.left - self.baseline_l)
        self.baseline_r += alpha * (reading.right - self.baseline_r)
        if not (self.active or self.holding) and abs(delta) < 3.0 * max(self.spread, _MIN_SPREAD):
            spread_alpha = ema_alpha(dt, 1.0 if calibrating else 6.0)
            self.spread = max(_MIN_SPREAD, self.spread + spread_alpha * (abs(delta) - self.spread))
        return event, activated


@dataclass
class _Cluster:
    start: float
    members: set[str] = field(default_factory=set)
    events: list[AUEvent] = field(default_factory=list)
    last_close: float = 0.0
    speaking: bool = False


def count_swings(values: list[float], min_swing: float) -> int:
    """Number of direction reversals of at least ``min_swing`` in a series."""
    if len(values) < 3:
        return 0
    swings = 0
    direction = 0
    low = high = extreme = values[0]
    for value in values[1:]:
        if direction == 0:
            low, high = min(low, value), max(high, value)
            if value - low >= min_swing:
                direction, extreme = 1, value
            elif high - value >= min_swing:
                direction, extreme = -1, value
        elif direction == 1:
            if value > extreme:
                extreme = value
            elif extreme - value >= min_swing:
                swings, direction, extreme = swings + 1, -1, value
        else:
            if value < extreme:
                extreme = value
            elif value - extreme >= min_swing:
                swings, direction, extreme = swings + 1, 1, value
    return swings


def _articulation_event(event: AUEvent) -> bool:
    """Lower-face movement during speech that is too brief or weak to be read as affect."""
    if AU_BY_CODE[event.code].region not in _LOWER_FACE_REGIONS:
        return False
    sustained = event.offset - event.onset > MICRO_MAX_SEC
    return not (sustained and event.apex_level >= _SPEAKING_LOWER_FACE_MIN_LEVEL)


def head_angles(matrix: Any) -> tuple[float, float, float] | None:
    """(pitch, yaw, roll) in degrees from MediaPipe's facial transformation matrix."""
    try:
        rot = np.asarray(matrix, dtype=np.float64)[:3, :3]
    except (TypeError, ValueError, IndexError):
        return None
    if rot.shape != (3, 3):
        return None
    sy = math.hypot(rot[0, 0], rot[1, 0])
    pitch = math.degrees(math.atan2(rot[2, 1], rot[2, 2]))
    yaw = math.degrees(math.atan2(-rot[2, 0], sy))
    roll = math.degrees(math.atan2(rot[1, 0], rot[0, 0]))
    return pitch, yaw, roll


class _Subject:
    def __init__(self, subject_id: str, settings: Settings, t: float, media_sec: float) -> None:
        self.id = subject_id
        self._settings = settings
        self.channels = {au.code: _AUChannel(au.code) for au in ACTION_UNITS}
        self.blend_filter = OneEuroFilter(min_cutoff=2.5, beta=4.0, d_cutoff=1.0)
        self.point_filter = OneEuroFilter(min_cutoff=1.2, beta=30.0, d_cutoff=1.0)
        self.angle_filter = OneEuroFilter(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
        self.pulse = PulseEstimator()
        self.first_seen = media_sec
        self.last_seen = media_sec
        self.last_t = t
        self.observed_sec = 0.0
        self.center = np.zeros(2, dtype=np.float32)
        self.face_w = 0.0
        self.points_norm: np.ndarray | None = None
        self.clusters: dict[int, _Cluster] = {}
        self._cluster_seq = 0

        self.activations: dict[str, float] = {}
        self.emotions: dict[str, float] = {}
        self.valence = 0.0
        self.activation = 0.0
        self.regions = dict.fromkeys(REGIONS, 0.0)

        self.blink_closed = False
        self.blink_start = 0.0
        self.blink_end = -1.0
        self.blinks: deque[float] = deque()
        self.blink_total = 0
        self.long_closures = 0
        self.last_burst = -10.0

        self.gaze_away = False
        self.gaze_start = 0.0
        self.gaze_dir = ""
        self.gaze_away_sec = 0.0

        self.angles: tuple[float, float, float] | None = None
        self.head_speed = 0.0
        self.angle_history: deque[tuple[float, float, float]] = deque()
        self.last_nod = -10.0
        self.last_shake = -10.0
        self.nods = 0
        self.shakes = 0

        self.jaw_history: deque[tuple[float, float]] = deque()
        self.speaking_until = -1.0

        self.touching = False
        self.touch_start = 0.0
        self.touch_zone = ""
        self.self_touch = 0

        self.crops: deque[tuple[float, np.ndarray]] = deque()
        self.last_crop_t = -1.0

        self.counts = {"micro": 0, "macro": 0, "behavior": 0}
        self.emotion_counts: dict[str, dict[str, int]] = {}
        self.cue_counts: dict[str, int] = {}
        self.time_by_emotion: dict[str, float] = {}
        self.valence_sum = 0.0
        self.activation_sum = 0.0
        self.affect_samples = 0
        self.pulse_samples: list[float] = []

        self.live_timeline: deque[tuple[float, float, float]] = deque(
            maxlen=_LIVE_TIMELINE_POINTS
        )
        self.last_live_sample = -1.0
        self.timeline: list[dict[str, Any]] = []
        self.bucket_start = t
        self.bucket: dict[str, float] = {}
        self.bucket_media = media_sec

        self.flash: str | None = None
        self.flash_at = -10.0

    @property
    def calibrated(self) -> bool:
        return self.observed_sec >= self._settings.expression_calibration_sec

    @property
    def speaking(self) -> bool:
        return self.last_t <= self.speaking_until

    def blinking(self, t: float) -> bool:
        return self.blink_closed or t - self.blink_end < 0.15

    def blink_rate(self, t: float) -> float:
        while self.blinks and t - self.blinks[0] > 60.0:
            self.blinks.popleft()
        window = min(60.0, max(self.observed_sec, 1.0))
        return round(len(self.blinks) * 60.0 / window, 1)

    def blink_baseline(self) -> float:
        return round(self.blink_total * 60.0 / max(self.observed_sec, 1.0), 1)

    def cancel_open_states(self) -> None:
        for channel in self.channels.values():
            channel.cancel()
        self.clusters.clear()
        self.blink_closed = False
        self.gaze_away = False
        self.touching = False
        self.angle_history.clear()
        self.jaw_history.clear()
        self.point_filter.reset()
        self.blend_filter.reset()
        self.angle_filter.reset()
        self.pulse.reset()

    # -- per-frame update -------------------------------------------------

    def update(
        self,
        t: float,
        media_sec: float,
        points_norm: np.ndarray,
        scores: dict[str, float],
        matrix: Any,
        frame_bgr: np.ndarray | None,
        hands: list[np.ndarray],
        *,
        primary: bool,
        capture_crops: bool,
        make_episode: Any,
    ) -> list[Episode]:
        dt = t - self.last_t
        if dt > _GAP_RESET_SEC or dt < 0:
            self.cancel_open_states()
            dt = 1.0 / 30.0
        dt = max(dt, 1e-3)
        self.last_t = t
        self.last_seen = media_sec
        self.observed_sec += dt
        self.points_norm = self.point_filter(points_norm, t)
        mins = points_norm.min(axis=0)
        maxs = points_norm.max(axis=0)
        self.center = (mins + maxs) * 0.5
        self.face_w = float(maxs[0] - mins[0])

        episodes: list[Episode] = []
        self._update_head(t, matrix)
        self._update_speech(t, scores.get("jawOpen", 0.0))
        episodes.extend(self._update_blink(t, media_sec, scores, make_episode))
        episodes.extend(self._update_gaze(t, media_sec, scores, make_episode))
        episodes.extend(self._update_touch(t, media_sec, hands, make_episode))
        episodes.extend(self._update_head_gestures(t, media_sec, make_episode))

        names = sorted(scores)
        raw = np.array([scores[name] for name in names], dtype=np.float32)
        smoothed = dict(zip(names, self.blend_filter(raw, t).tolist()))
        readings = read_action_units(smoothed)
        episodes.extend(self._update_channels(t, media_sec, dt, readings, make_episode))
        self._update_affect(t, media_sec, dt, readings)

        if frame_bgr is not None:
            height, width = frame_bgr.shape[:2]
            points_px = self.points_norm * np.array([width, height], dtype=np.float32)
            if primary and self._settings.expression_pulse_enabled and self.calibrated:
                reading = self.pulse.add(
                    t, skin_rgb(frame_bgr, points_px), moving=self.head_speed > 30.0
                )
                if reading is not None and reading.confidence >= 0.5:
                    self.bucket["bpm"] = reading.bpm
            if capture_crops and t - self.last_crop_t >= _CROP_INTERVAL_SEC:
                self._store_crop(t, frame_bgr, points_px)
        return episodes

    def _update_head(self, t: float, matrix: Any) -> None:
        angles = head_angles(matrix) if matrix is not None else None
        if angles is None:
            return
        filtered = self.angle_filter(np.array(angles, dtype=np.float32), t)
        prev = self.angles
        self.angles = (float(filtered[0]), float(filtered[1]), float(filtered[2]))
        if prev is not None:
            dt = max(t - self.angle_history[-1][0], 1e-3) if self.angle_history else 1.0 / 30
            speed = math.dist(prev, self.angles) / dt
            self.head_speed += ema_alpha(dt, 0.3) * (speed - self.head_speed)
        self.angle_history.append((t, self.angles[0], self.angles[1]))
        while self.angle_history and t - self.angle_history[0][0] > 1.6:
            self.angle_history.popleft()

    def _update_speech(self, t: float, jaw: float) -> None:
        self.jaw_history.append((t, jaw))
        while self.jaw_history and t - self.jaw_history[0][0] > 1.5:
            self.jaw_history.popleft()
        if len(self.jaw_history) < 6:
            return
        values = [v for _, v in self.jaw_history]
        if float(np.std(values)) > 0.035 and count_swings(values, 0.06) >= 3:
            self.speaking_until = t + 0.6

    def _update_blink(
        self, t: float, media_sec: float, scores: dict[str, float], make_episode: Any
    ) -> list[Episode]:
        closure = (scores.get("eyeBlinkLeft", 0.0) + scores.get("eyeBlinkRight", 0.0)) / 2.0
        out: list[Episode] = []
        if not self.blink_closed and closure > 0.5:
            self.blink_closed = True
            self.blink_start = t
        elif self.blink_closed and closure < 0.32:
            self.blink_closed = False
            self.blink_end = t
            duration = t - self.blink_start
            if duration <= 0.5:
                self.blinks.append(t)
                self.blink_total += 1
                recent = [b for b in self.blinks if t - b <= 2.5]
                if len(recent) >= 3 and t - self.last_burst > 3.0:
                    self.last_burst = t
                    out.append(
                        make_episode(
                            self, "blink_burst", media_sec - 2.5, t, 2500, len(recent) / 5.0,
                            detail=f"{len(recent)} blinks",
                        )
                    )
            else:
                self.long_closures += 1
                out.append(
                    make_episode(
                        self, "eye_closure", media_sec - duration, t, int(duration * 1000),
                        min(1.0, duration / 2.0),
                    )
                )
        return out

    def _update_gaze(
        self, t: float, media_sec: float, scores: dict[str, float], make_episode: Any
    ) -> list[Episode]:
        if self.blinking(t):
            return []
        looks = {
            "left": (scores.get("eyeLookOutLeft", 0.0) + scores.get("eyeLookInRight", 0.0)) / 2,
            "right": (scores.get("eyeLookInLeft", 0.0) + scores.get("eyeLookOutRight", 0.0)) / 2,
            "up": (scores.get("eyeLookUpLeft", 0.0) + scores.get("eyeLookUpRight", 0.0)) / 2,
            "down": 0.85
            * (scores.get("eyeLookDownLeft", 0.0) + scores.get("eyeLookDownRight", 0.0))
            / 2,
        }
        direction, amount = max(looks.items(), key=lambda item: item[1])
        if self.angles is not None and abs(self.angles[1]) > 30.0:
            direction, amount = ("left" if self.angles[1] > 0 else "right"), 1.0
        out: list[Episode] = []
        if not self.gaze_away and amount > 0.55:
            self.gaze_away = True
            self.gaze_start = t
            self.gaze_dir = direction
        elif self.gaze_away and amount < 0.4:
            self.gaze_away = False
            duration = t - self.gaze_start
            self.gaze_away_sec += duration
            if duration >= 1.0:
                out.append(
                    make_episode(
                        self, "gaze_aversion", media_sec - duration, t, int(duration * 1000),
                        min(1.0, duration / 4.0), detail=self.gaze_dir,
                    )
                )
        return out

    def _update_touch(
        self, t: float, media_sec: float, hands: list[np.ndarray], make_episode: Any
    ) -> list[Episode]:
        if self.points_norm is None:
            return []
        mins = self.points_norm.min(axis=0)
        maxs = self.points_norm.max(axis=0)
        pad = (maxs - mins) * 0.12
        lo, hi = mins - pad, maxs + pad
        inside_y: list[float] = []
        for hand in hands:
            mask = np.all((hand >= lo) & (hand <= hi), axis=1)
            if int(mask.sum()) >= 4:
                inside_y.extend(((hand[mask, 1] - mins[1]) / max(1e-6, maxs[1] - mins[1])).tolist())
        out: list[Episode] = []
        if inside_y and not self.touching:
            self.touching = True
            self.touch_start = t
            mean_y = float(np.mean(inside_y))
            self.touch_zone = (
                "forehead / eyes" if mean_y < 0.35 else "nose / cheek" if mean_y < 0.65
                else "mouth / chin"
            )
        elif not inside_y and self.touching:
            self.touching = False
            duration = t - self.touch_start
            if duration >= 0.4:
                self.self_touch += 1
                out.append(
                    make_episode(
                        self, "self_touch", media_sec - duration, t, int(duration * 1000),
                        min(1.0, duration / 3.0), detail=self.touch_zone,
                    )
                )
        return out

    def _update_head_gestures(
        self, t: float, media_sec: float, make_episode: Any
    ) -> list[Episode]:
        if len(self.angle_history) < 8:
            return []
        pitches = [p for _, p, _ in self.angle_history]
        yaws = [y for _, _, y in self.angle_history]
        pitch_range = max(pitches) - min(pitches)
        yaw_range = max(yaws) - min(yaws)
        out: list[Episode] = []
        if (
            t - self.last_nod > 2.0
            and pitch_range > 5.0
            and yaw_range < 0.6 * pitch_range
            and count_swings(pitches, 3.5) >= 2
        ):
            self.last_nod = t
            self.nods += 1
            out.append(make_episode(self, "head_nod", media_sec - 1.2, t, 1200, pitch_range / 20))
        if (
            t - self.last_shake > 2.0
            and yaw_range > 6.0
            and pitch_range < 0.6 * yaw_range
            and count_swings(yaws, 4.0) >= 2
        ):
            self.last_shake = t
            self.shakes += 1
            out.append(make_episode(self, "head_shake", media_sec - 1.2, t, 1200, yaw_range / 25))
        return out

    def _update_channels(
        self,
        t: float,
        media_sec: float,
        dt: float,
        readings: dict[str, AUReading],
        make_episode: Any,
    ) -> list[Episode]:
        calibrating = not self.calibrated
        speaking = self.speaking
        blinking = self.blinking(t)
        head_fast = self.head_speed > _HEAD_FAST_DEG_S
        sensitivity = self._settings.expression_sensitivity
        for code, channel in self.channels.items():
            au = channel.au
            allow = not head_fast
            if speaking and au.speech_sensitive:
                allow = False
            if blinking and code in _EYE_AUS:
                allow = False
            scale = 1.6 if speaking and au.region in _LOWER_FACE_REGIONS else 1.0
            event, activated = channel.step(
                readings[code],
                t,
                dt,
                calibrating=calibrating,
                allow_onset=allow,
                threshold_scale=scale,
                sensitivity=sensitivity,
            )
            if activated:
                self._join_cluster(channel, speaking)
            if event is not None and channel.cluster is not None:
                cluster = self.clusters.get(channel.cluster)
                if cluster is not None:
                    cluster.events.append(event)
                    cluster.members.discard(code)
                    cluster.last_close = t
                channel.cluster = None
            elif channel.cluster is not None and not channel.active:
                cluster = self.clusters.get(channel.cluster)
                if cluster is not None:
                    cluster.members.discard(code)
                    cluster.last_close = t
                channel.cluster = None
            self.activations[code] = channel.level(readings[code].value)

        out: list[Episode] = []
        for key in list(self.clusters):
            cluster = self.clusters[key]
            if any(is_negative_event(e) for e in cluster.events):
                grace = _NEGATIVE_CLUSTER_GRACE_SEC
            elif cluster.events and all(
                AU_BY_CODE[e.code].speech_sensitive for e in cluster.events
            ):
                grace = _SPEECH_ONSET_GRACE_SEC
            else:
                grace = _CLUSTER_GRACE_SEC
            if cluster.members or t - cluster.last_close < grace:
                continue
            del self.clusters[key]
            cluster.speaking = cluster.speaking or speaking
            if cluster.events:
                episode = self._finalize_cluster(cluster, t, media_sec, make_episode)
                if episode is not None:
                    out.append(episode)
        return out

    def _join_cluster(self, channel: _AUChannel, speaking: bool) -> None:
        for key, cluster in self.clusters.items():
            if abs(channel.onset - cluster.start) <= _CLUSTER_WINDOW_SEC:
                cluster.members.add(channel.au.code)
                channel.cluster = key
                return
        self._cluster_seq += 1
        self.clusters[self._cluster_seq] = _Cluster(
            start=channel.onset, members={channel.au.code}, speaking=speaking
        )
        channel.cluster = self._cluster_seq

    def _finalize_cluster(
        self, cluster: _Cluster, t: float, media_sec: float, make_episode: Any
    ) -> Episode | None:
        speaking = cluster.speaking
        events = [
            e
            for e in cluster.events
            if not (speaking and _articulation_event(e))
        ]
        if not events:
            return None
        onset = min(e.onset for e in events)
        offset = max(e.offset for e in events)
        duration = max(0.0, offset - onset)
        strongest = max(events, key=lambda e: e.apex_level)
        codes = {e.code for e in events}
        if strongest.apex_level < _MIN_EPISODE_LEVEL and len(codes) == 1:
            return None
        baton = speaking and codes <= _BROW_AUS and duration <= _BATON_MAX_SEC

        def baton_episode(detail: str) -> Episode:
            return make_episode(
                self, "speech_emphasis", media_sec - (t - onset), onset,
                int(duration * 1000), strongest.apex_level, detail=detail, speaking=True,
                aus=sorted(codes, key=lambda c: int(c[2:])),
            )

        if baton and codes <= _BROW_RAISE_AUS:
            return baton_episode("brow raise")

        act = {e.code: e.apex_level for e in events}
        left = {e.code: e.left if e.side else e.apex_level for e in events}
        right = {e.code: e.right if e.side else e.apex_level for e in events}
        for code, channel in self.channels.items():
            if code in act or not (channel.active or channel.holding):
                continue
            if speaking and channel.au.region in _LOWER_FACE_REGIONS:
                continue
            context = self.activations.get(code, 0.0) * 0.7
            act[code] = left[code] = right[code] = context
        emotions = score_emotions(act, left, right)
        emotion, score = dominant_emotion(emotions, floor=0.25)
        if emotion != "neutral" and not emotion_involves(emotion, codes):
            # Context AUs alone can't name the episode (an AU18 pucker during a frown isn't anger).
            emotion, score = "neutral", 0.0
        if baton and emotion == "neutral":
            return baton_episode("brow lower" if "AU4" in codes else "brow")

        smile = self.channels["AU12"]
        smile_onset = next((e.onset for e in events if e.code == "AU12"), None)
        if smile_onset is None and (smile.active or smile.holding):
            smile_onset = smile.onset
        press = self.channels["AU24"]
        press_onset = next((e.onset for e in events if e.code in {"AU24", "AU17"}), None)
        if press_onset is None and press.active:
            press_onset = press.onset
        cues = episode_cues(events, onset, duration, smile_onset, press_onset)
        if speaking:
            cues = [cue for cue in cues if cue not in _SPEECH_CONFOUNDED_CUES]
            lower_only = all(AU_BY_CODE[code].region in _LOWER_FACE_REGIONS for code in codes)
            if lower_only and emotion == "neutral" and not cues:
                return None

        kind = "micro" if duration <= MICRO_MAX_SEC else "macro"
        label = (
            EMOTION_LABELS[emotion] if emotion != "neutral" else AU_BY_CODE[strongest.code].name
        )
        sides = {e.side for e in events if e.side}
        side = strongest.side if len(sides) == 1 or strongest.code in {"AU12", "AU14"} else ""
        episode = make_episode(
            self,
            None,
            media_sec - (t - onset),
            onset,
            int(duration * 1000),
            strongest.apex_level,
            kind=kind,
            label=label,
            aus=sorted({e.code for e in events}, key=lambda c: int(c[2:])),
            emotion=None if emotion == "neutral" else emotion,
            emotion_score=score if emotion != "neutral" else 0.0,
            side=side if side in {"L", "R"} else ("bilateral" if side else ""),
            cues=cues,
            speaking=cluster.speaking,
            apex_t=strongest.apex_t,
        )
        if kind == "micro" or episode.notable:
            self.flash = f"{'MICRO' if kind == 'micro' else 'EXPR'} | {label.upper()}"
            self.flash_at = t
        return episode

    def _update_affect(
        self, t: float, media_sec: float, dt: float, readings: dict[str, AUReading]
    ) -> None:
        if not self.calibrated:
            self.emotions = {}
            self.valence = 0.0
            self.activation = 0.0
            self.regions = dict.fromkeys(REGIONS, 0.0)
            return
        left = {
            code: ch.side_level(readings[code].left, True) for code, ch in self.channels.items()
        }
        right = {
            code: ch.side_level(readings[code].right, False) for code, ch in self.channels.items()
        }
        self.emotions = score_emotions(self.activations, left, right)
        self.valence = valence(self.emotions)
        self.activation = activation(self.activations)
        self.regions = region_levels(self.activations)
        dominant, _ = dominant_emotion(self.emotions)
        self.time_by_emotion[dominant] = self.time_by_emotion.get(dominant, 0.0) + dt
        self.valence_sum += self.valence
        self.activation_sum += self.activation
        self.affect_samples += 1

        if t - self.last_live_sample >= _LIVE_TIMELINE_STEP_SEC:
            self.last_live_sample = t
            self.live_timeline.append((round(media_sec, 2), self.valence, self.activation))

        bucket = self.bucket
        bucket["v"] = bucket.get("v", 0.0) + self.valence
        bucket["a"] = bucket.get("a", 0.0) + self.activation
        bucket["n"] = bucket.get("n", 0.0) + 1.0
        if t - self.bucket_start >= 1.0:
            n = max(1.0, bucket["n"])
            entry: dict[str, Any] = {
                "t": round(self.bucket_media, 1),
                "valence": round(bucket["v"] / n, 3),
                "activation": round(bucket["a"] / n, 3),
                "emotion": dominant,
            }
            if "bpm" in bucket:
                entry["bpm"] = bucket["bpm"]
                self.pulse_samples.append(float(bucket["bpm"]))
            self.timeline.append(entry)
            self.bucket = {}
            self.bucket_start = t
            self.bucket_media = media_sec

    def _store_crop(self, t: float, frame_bgr: np.ndarray, points_px: np.ndarray) -> None:
        self.last_crop_t = t
        height, width = frame_bgr.shape[:2]
        x1, y1 = points_px.min(axis=0)
        x2, y2 = points_px.max(axis=0)
        pad_x, pad_y = (x2 - x1) * 0.18, (y2 - y1) * 0.18
        ix1, iy1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        ix2, iy2 = min(width, int(x2 + pad_x)), min(height, int(y2 + pad_y))
        if ix2 - ix1 < 24 or iy2 - iy1 < 24:
            return
        crop = frame_bgr[iy1:iy2, ix1:ix2]
        if crop.shape[1] > _CROP_MAX_WIDTH:
            scale = _CROP_MAX_WIDTH / crop.shape[1]
            crop = cv2.resize(
                crop, (_CROP_MAX_WIDTH, int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA
            )
        else:
            crop = crop.copy()
        self.crops.append((t, crop))
        while self.crops and t - self.crops[0][0] > _CROP_BUFFER_SEC:
            self.crops.popleft()

    def crop_near(self, t: float) -> np.ndarray | None:
        if not self.crops:
            return None
        return min(self.crops, key=lambda item: abs(item[0] - t))[1]

    def record(self, episode: Episode) -> None:
        self.counts[episode.kind] = self.counts.get(episode.kind, 0) + 1
        if episode.emotion:
            bucket = self.emotion_counts.setdefault(episode.emotion, {"micro": 0, "macro": 0})
            if episode.kind in bucket:
                bucket[episode.kind] += 1
        for cue in episode.cues:
            self.cue_counts[cue] = self.cue_counts.get(cue, 0) + 1

    # -- reporting --------------------------------------------------------

    def live_status(self, t: float, primary: bool) -> dict[str, Any]:
        dominant, dominant_score = dominant_emotion(self.emotions)
        top_aus = sorted(
            ((code, level) for code, level in self.activations.items() if level >= 0.12),
            key=lambda item: item[1],
            reverse=True,
        )[:6]
        reading = self.pulse.reading
        calibration = min(1.0, self.observed_sec / self._settings.expression_calibration_sec)
        return {
            "id": self.id,
            "primary": primary,
            "calibrated": self.calibrated,
            "calibration_pct": round(calibration * 100),
            "emotions": dict(self.emotions),
            "dominant": {
                "emotion": dominant,
                "label": EMOTION_LABELS.get(dominant, dominant),
                "score": round(dominant_score, 2),
            },
            "valence": self.valence,
            "activation": self.activation,
            "regions": {k: round(v, 3) for k, v in self.regions.items()},
            "aus": [
                {"code": code, "name": AU_BY_CODE[code].name, "level": round(level, 2)}
                for code, level in top_aus
            ],
            "blink_rate": self.blink_rate(t),
            "blink_baseline": self.blink_baseline(),
            "blink_total": self.blink_total,
            "gaze_away": self.gaze_away,
            "gaze_away_pct": round(100.0 * self.gaze_away_sec / max(self.observed_sec, 1.0), 1),
            "head": (
                {
                    "pitch": round(self.angles[0], 1),
                    "yaw": round(self.angles[1], 1),
                    "roll": round(self.angles[2], 1),
                    "speed": round(self.head_speed, 1),
                }
                if self.angles is not None
                else None
            ),
            "speaking": self.speaking,
            "self_touch": self.touching,
            "pulse": (
                {"bpm": reading.bpm, "confidence": reading.confidence}
                if reading is not None and reading.confidence >= _PULSE_SHOW_CONFIDENCE
                else None
            ),
            "counts": dict(self.counts),
        }

    def summary(self) -> dict[str, Any]:
        samples = max(1, self.affect_samples)
        pulse = None
        if self.pulse_samples:
            pulse = {
                "mean": round(float(np.mean(self.pulse_samples)), 1),
                "min": round(float(np.min(self.pulse_samples)), 1),
                "max": round(float(np.max(self.pulse_samples)), 1),
                "samples": len(self.pulse_samples),
            }
        return {
            "id": self.id,
            "first_seen": round(self.first_seen, 1),
            "last_seen": round(self.last_seen, 1),
            "observed_sec": round(self.observed_sec, 1),
            "calibrated": self.calibrated,
            "episodes": dict(self.counts),
            "emotions": {k: dict(v) for k, v in self.emotion_counts.items()},
            "cues": dict(sorted(self.cue_counts.items(), key=lambda item: -item[1])),
            "blink": {
                "total": self.blink_total,
                "rate_per_min": self.blink_baseline(),
                "long_closures": self.long_closures,
            },
            "gaze_away_pct": round(100.0 * self.gaze_away_sec / max(self.observed_sec, 1.0), 1),
            "head_nods": self.nods,
            "head_shakes": self.shakes,
            "self_touch": self.self_touch,
            "pulse": pulse,
            "affect": {
                "valence_mean": round(self.valence_sum / samples, 3),
                "activation_mean": round(self.activation_sum / samples, 3),
                "time_by_emotion_sec": {
                    k: round(v, 1)
                    for k, v in sorted(self.time_by_emotion.items(), key=lambda item: -item[1])
                },
            },
        }


def _idle_status(enabled: bool = False, state: str = "off") -> dict[str, Any]:
    return {
        "enabled": enabled,
        "state": state,
        "subjects": [],
        "episodes": [],
        "timeline": [],
        "counts": {"micro": 0, "macro": 0, "behavior": 0},
        "events": [],
    }


class ExpressionAnalyzer:
    """Tracks every face across frames and exposes live status + session analysis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self.capture_crops = False
        self._subjects: dict[str, _Subject] = {}
        self._episodes: list[Episode] = []
        self._episode_seq = 0
        self._primary: str | None = None
        self._last_t = 0.0
        self._enabled = False
        self._status: dict[str, Any] = _idle_status()
        self._status_at = -1.0

    def reset(self) -> None:
        with self._lock:
            self._subjects.clear()
            self._episodes.clear()
            self._episode_seq = 0
            self._primary = None
            self._last_t = 0.0
            self._status = _idle_status(self._enabled, "idle" if self._enabled else "off")
            self._status_at = -1.0

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled
            if not enabled:
                for subject in self._subjects.values():
                    subject.cancel_open_states()
            self._status_at = -1.0

    def _make_episode(
        self,
        subject: _Subject,
        cue_key: str | None,
        at: float,
        t: float,
        duration_ms: int,
        intensity: float,
        *,
        detail: str = "",
        kind: str = "behavior",
        label: str = "",
        aus: list[str] | None = None,
        emotion: str | None = None,
        emotion_score: float = 0.0,
        side: str = "",
        cues: list[str] | None = None,
        speaking: bool = False,
        apex_t: float | None = None,
    ) -> Episode:
        self._episode_seq += 1
        if cue_key is not None:
            cue = CUES[cue_key]
            label = f"{cue.label} · {detail}" if detail else cue.label
            cues = [cue_key]
        episode = Episode(
            id=f"E{self._episode_seq:04d}",
            subject=subject.id,
            kind=kind,
            label=label,
            at=max(0.0, at),
            t=t,
            duration_ms=duration_ms,
            intensity=max(0.0, min(1.0, intensity)),
            aus=aus or [],
            emotion=emotion,
            emotion_score=emotion_score,
            side=side,
            cues=cues or [],
            speaking=speaking,
        )
        if self.capture_crops and kind != "behavior":
            episode.apex_crop = subject.crop_near(apex_t if apex_t is not None else t)
        subject.record(episode)
        return episode

    def _assign_subjects(
        self, faces: list[np.ndarray], t: float, media_sec: float
    ) -> list[_Subject]:
        candidates = [
            s for s in self._subjects.values() if t - s.last_t <= _SUBJECT_REACQUIRE_SEC
        ]
        assigned: list[_Subject | None] = [None] * len(faces)
        centers = [(f.min(axis=0) + f.max(axis=0)) * 0.5 for f in faces]
        widths = [float(f[:, 0].max() - f[:, 0].min()) for f in faces]
        pairs = sorted(
            (
                (float(np.linalg.norm(centers[i] - s.center)), i, s)
                for i in range(len(faces))
                for s in candidates
            ),
            key=lambda item: item[0],
        )
        used: set[str] = set()
        for distance, index, subject in pairs:
            if assigned[index] is not None or subject.id in used:
                continue
            recent = t - subject.last_t <= _SUBJECT_TIMEOUT_SEC
            radius = max(0.12, 0.9 * widths[index]) * (1.0 if recent else 1.6)
            if distance > radius:
                continue
            assigned[index] = subject
            used.add(subject.id)
        for index, subject in enumerate(assigned):
            if subject is None:
                name = _SUBJECT_NAMES[len(self._subjects) % len(_SUBJECT_NAMES)]
                if name in self._subjects:
                    name = f"{name}{len(self._subjects)}"
                subject = _Subject(name, self._settings, t, media_sec)
                self._subjects[name] = subject
                assigned[index] = subject
        return [s for s in assigned if s is not None]

    def process(
        self,
        face_result: Any,
        frame_bgr: np.ndarray | None,
        t: float,
        media_sec: float,
        hand_points: list[np.ndarray] | None = None,
    ) -> ExpressionFrame:
        landmarks = getattr(face_result, "face_landmarks", None) or []
        blendshapes = getattr(face_result, "face_blendshapes", None) or []
        matrices = getattr(face_result, "facial_transformation_matrixes", None) or []
        hands = hand_points or []
        with self._lock:
            self._last_t = t
            if not landmarks or not blendshapes:
                return ExpressionFrame(renders=[], episodes=[])
            faces = [
                np.array([[lm.x, lm.y] for lm in face], dtype=np.float32) for face in landmarks
            ]
            subjects = self._assign_subjects(faces, t, media_sec)
            widths = [float(f[:, 0].max() - f[:, 0].min()) for f in faces]
            primary_index = int(np.argmax(widths))
            self._primary = subjects[primary_index].id

            new_episodes: list[Episode] = []
            renders: list[SubjectRender] = []
            for index, (subject, points) in enumerate(zip(subjects, faces)):
                scores = blendshape_scores(blendshapes[index]) if index < len(blendshapes) else {}
                if not scores:
                    continue
                matrix = matrices[index] if index < len(matrices) else None
                episodes = subject.update(
                    t,
                    media_sec,
                    points,
                    scores,
                    matrix,
                    frame_bgr,
                    hands,
                    primary=index == primary_index,
                    capture_crops=self.capture_crops,
                    make_episode=self._make_episode,
                )
                new_episodes.extend(episodes)
                renders.append(self._render(subject, t, frame_bgr, index == primary_index))

            if new_episodes:
                self._episodes.extend(new_episodes)
                if len(self._episodes) > _MAX_EPISODES:
                    del self._episodes[: len(self._episodes) - _MAX_EPISODES]
            return ExpressionFrame(renders=renders, episodes=new_episodes)

    def _render(
        self, subject: _Subject, t: float, frame_bgr: np.ndarray | None, primary: bool
    ) -> SubjectRender:
        if frame_bgr is not None and subject.points_norm is not None:
            height, width = frame_bgr.shape[:2]
            points_px = subject.points_norm * np.array([width, height], dtype=np.float32)
        else:
            points_px = np.zeros((0, 2), dtype=np.float32)
        dominant, score = dominant_emotion(subject.emotions)
        if not subject.calibrated:
            calibration = min(1.0, subject.observed_sec / self._settings.expression_calibration_sec)
            tag = f"{subject.id} | CALIBRATING {calibration:.0%}"
        elif dominant == "neutral":
            tag = f"{subject.id} | NEUTRAL"
        else:
            tag = f"{subject.id} | {EMOTION_LABELS[dominant].upper()} {score:.0%}"
        age = t - subject.flash_at
        flash = subject.flash if age <= _FLASH_SEC else None
        return SubjectRender(
            subject=subject.id,
            points_px=points_px,
            regions=dict(subject.regions),
            tag=tag,
            flash=flash,
            flash_strength=max(0.0, 1.0 - age / _FLASH_SEC) if flash else 0.0,
            primary=primary,
        )

    def status(self) -> dict[str, Any]:
        with self._lock:
            if not self._enabled:
                return _idle_status(False, "off")
            if self._status_at >= 0 and self._last_t - self._status_at < _STATUS_INTERVAL_SEC:
                return self._status
            self._status_at = self._last_t
            self._status = self._build_status()
            return self._status

    def _build_status(self) -> dict[str, Any]:
        t = self._last_t
        active = [s for s in self._subjects.values() if t - s.last_t <= _SUBJECT_TIMEOUT_SEC]
        if not active:
            status = _idle_status(True, "no_face")
        else:
            active.sort(key=lambda s: (s.id != self._primary, s.id))
            primary = active[0]
            status = _idle_status(True, "live" if primary.calibrated else "calibrating")
            status["subjects"] = [s.live_status(t, s.id == self._primary) for s in active]
            status["timeline"] = [
                {"t": at, "v": round(v, 3), "a": round(a, 3)} for at, v, a in primary.live_timeline
            ]
            status["events"] = [
                {
                    "label": f"{AU_BY_CODE[item['code']].name} ({item['code']})",
                    "region": AU_BY_CODE[item["code"]].region,
                    "intensity": item["level"],
                    "ts": t,
                }
                for item in status["subjects"][0]["aus"]
            ]
        status["episodes"] = [e.as_dict() for e in reversed(self._episodes[-14:])]
        counts = {"micro": 0, "macro": 0, "behavior": 0}
        for subject in self._subjects.values():
            for kind, value in subject.counts.items():
                counts[kind] = counts.get(kind, 0) + value
        status["counts"] = counts
        return status

    def summary(self) -> dict[str, Any]:
        """Compact per-subject aggregates (used for insight digests and reports)."""
        with self._lock:
            return {
                "subjects": [s.summary() for s in self._subjects.values()],
                "primary": self._primary,
            }

    def analysis(self, max_episodes: int = _MAX_EPISODES) -> dict[str, Any]:
        with self._lock:
            episodes = self._episodes[-max_episodes:]
            cue_counts: dict[str, int] = {}
            counts = {"micro": 0, "macro": 0, "behavior": 0}
            for subject in self._subjects.values():
                for cue, value in subject.cue_counts.items():
                    cue_counts[cue] = cue_counts.get(cue, 0) + value
                for kind, value in subject.counts.items():
                    counts[kind] = counts.get(kind, 0) + value
            return {
                "subjects": [s.summary() for s in self._subjects.values()],
                "episodes": [e.as_dict() for e in episodes],
                "timeline": {s.id: list(s.timeline) for s in self._subjects.values()},
                "counts": counts,
                "cue_counts": dict(sorted(cue_counts.items(), key=lambda item: -item[1])),
                "cue_meanings": {key: CUES[key].meaning for key in cue_counts if key in CUES},
            }
