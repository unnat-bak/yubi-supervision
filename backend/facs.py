"""FACS action units from face blendshapes, EMFACS emotion prototypes, and cue library.

MediaPipe's face landmarker emits 52 ARKit-style blendshapes. They map closely
onto Facial Action Coding System (FACS) action units, which lets the engine
reason in the vocabulary used by facial-expression research: an episode is a
set of AUs with onset/apex/offset timing, and emotions are recognised from
EMFACS prototype combinations (e.g. Duchenne joy = AU6 + AU12).

Everything here is pure and frame-rate independent so it can be unit-tested.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class ActionUnit:
    code: str
    name: str
    region: str
    left: str | None = None
    right: str | None = None
    bilateral: tuple[str, ...] = ()
    # Blendshape delta above the subject's baseline that reads as full intensity.
    gain: float = 0.45
    # Minimum rise above baseline before an episode can open.
    min_delta: float = 0.07
    # Moves with speech articulation; ignored for episodes while the subject talks.
    speech_sensitive: bool = False

    @property
    def lateral(self) -> bool:
        return self.left is not None and self.right is not None


ACTION_UNITS: tuple[ActionUnit, ...] = (
    ActionUnit("AU1", "Inner brow raiser", "brow", bilateral=("browInnerUp",), gain=0.4),
    ActionUnit(
        "AU2", "Outer brow raiser", "brow",
        left="browOuterUpLeft", right="browOuterUpRight", gain=0.4,
    ),
    ActionUnit(
        "AU4", "Brow lowerer", "brow",
        left="browDownLeft", right="browDownRight", gain=0.4,
    ),
    ActionUnit(
        "AU5", "Upper lid raiser", "eyes",
        left="eyeWideLeft", right="eyeWideRight", gain=0.35,
    ),
    ActionUnit(
        "AU6", "Cheek raiser", "eyes",
        left="cheekSquintLeft", right="cheekSquintRight", gain=0.35, min_delta=0.06,
    ),
    ActionUnit(
        "AU7", "Lid tightener", "eyes",
        left="eyeSquintLeft", right="eyeSquintRight", gain=0.4,
    ),
    ActionUnit(
        "AU9", "Nose wrinkler", "nose",
        left="noseSneerLeft", right="noseSneerRight", gain=0.3, min_delta=0.06,
    ),
    ActionUnit(
        "AU10", "Upper lip raiser", "nose",
        left="mouthUpperUpLeft", right="mouthUpperUpRight", gain=0.4,
    ),
    ActionUnit(
        "AU12", "Lip corner puller", "mouth",
        left="mouthSmileLeft", right="mouthSmileRight", gain=0.5, min_delta=0.08,
    ),
    ActionUnit(
        "AU14", "Dimpler", "mouth",
        left="mouthDimpleLeft", right="mouthDimpleRight", gain=0.35, min_delta=0.06,
    ),
    ActionUnit(
        "AU15", "Lip corner depressor", "mouth",
        left="mouthFrownLeft", right="mouthFrownRight", gain=0.35, min_delta=0.06,
    ),
    ActionUnit(
        "AU16", "Lower lip depressor", "mouth",
        left="mouthLowerDownLeft", right="mouthLowerDownRight", speech_sensitive=True,
    ),
    ActionUnit("AU17", "Chin raiser", "mouth", bilateral=("mouthShrugLower",), gain=0.4),
    ActionUnit(
        "AU18", "Lip pucker", "mouth", bilateral=("mouthPucker",), speech_sensitive=True,
    ),
    ActionUnit(
        "AU20", "Lip stretcher", "mouth",
        left="mouthStretchLeft", right="mouthStretchRight", gain=0.35, min_delta=0.06,
    ),
    ActionUnit(
        "AU22", "Lip funneler", "mouth", bilateral=("mouthFunnel",), speech_sensitive=True,
    ),
    ActionUnit(
        "AU24", "Lip pressor", "mouth",
        left="mouthPressLeft", right="mouthPressRight", gain=0.4, min_delta=0.07,
    ),
    ActionUnit(
        "AU26", "Jaw drop", "jaw", bilateral=("jawOpen",), gain=0.55, speech_sensitive=True,
    ),
    ActionUnit(
        "AU28", "Lip suck", "mouth",
        bilateral=("mouthRollLower", "mouthRollUpper"), gain=0.45, speech_sensitive=True,
    ),
    ActionUnit("AU29", "Jaw thrust", "jaw", bilateral=("jawForward",), gain=0.35),
    ActionUnit("AU30", "Jaw sideways", "jaw", bilateral=("jawLeft", "jawRight"), gain=0.35),
    ActionUnit("AU34", "Cheek puff", "cheek", bilateral=("cheekPuff",), gain=0.4),
)

AU_BY_CODE: dict[str, ActionUnit] = {au.code: au for au in ACTION_UNITS}
REGIONS: tuple[str, ...] = ("brow", "eyes", "nose", "mouth")
_REGION_OF_AU: dict[str, str] = {
    au.code: ("mouth" if au.region in {"jaw", "cheek"} else au.region) for au in ACTION_UNITS
}


@dataclass(frozen=True)
class AUReading:
    value: float
    left: float
    right: float


def blendshape_scores(categories: Iterable[object]) -> dict[str, float]:
    """MediaPipe ``Category`` list (or (name, score) pairs) -> {name: score}."""
    scores: dict[str, float] = {}
    for item in categories:
        if isinstance(item, tuple):
            name, score = item
        else:
            name = getattr(item, "category_name", None) or getattr(item, "display_name", "")
            score = getattr(item, "score", 0.0)
        if name:
            scores[str(name)] = float(score)
    return scores


def read_action_units(scores: Mapping[str, float]) -> dict[str, AUReading]:
    readings: dict[str, AUReading] = {}
    for au in ACTION_UNITS:
        if au.lateral:
            left = float(scores.get(au.left or "", 0.0))
            right = float(scores.get(au.right or "", 0.0))
            value = max(left, right)
        else:
            value = max((float(scores.get(name, 0.0)) for name in au.bilateral), default=0.0)
            left = right = value
        readings[au.code] = AUReading(value=value, left=left, right=right)
    return readings


def region_levels(activations: Mapping[str, float]) -> dict[str, float]:
    levels = dict.fromkeys(REGIONS, 0.0)
    for code, level in activations.items():
        region = _REGION_OF_AU.get(code)
        if region is not None:
            levels[region] = max(levels[region], float(level))
    return levels


@dataclass(frozen=True)
class EmotionPrototype:
    name: str
    label: str
    # Every group must be present (max over the AUs inside a group).
    core: tuple[tuple[str, ...], ...]
    support: tuple[str, ...]
    valence: float


EMOTION_PROTOTYPES: tuple[EmotionPrototype, ...] = (
    EmotionPrototype("happiness", "Joy", (("AU12",),), ("AU6",), 1.0),
    EmotionPrototype(
        "surprise", "Surprise", (("AU1", "AU2"), ("AU5", "AU26")), ("AU1", "AU2", "AU5"), 0.1
    ),
    EmotionPrototype(
        "fear", "Fear", (("AU1", "AU2"), ("AU4",), ("AU5", "AU20")), ("AU7", "AU20", "AU26"), -0.8
    ),
    EmotionPrototype(
        "anger", "Anger", (("AU4",), ("AU5", "AU7", "AU24")), ("AU5", "AU7", "AU24", "AU17"), -0.9
    ),
    EmotionPrototype(
        "disgust", "Disgust", (("AU9", "AU10"),), ("AU15", "AU16", "AU17", "AU4"), -0.8
    ),
    EmotionPrototype(
        "sadness", "Sadness", (("AU15",), ("AU1", "AU4")), ("AU1", "AU4", "AU17"), -0.7
    ),
)
EMOTIONS: tuple[str, ...] = tuple(p.name for p in EMOTION_PROTOTYPES) + ("contempt",)
EMOTION_LABELS: dict[str, str] = {p.name: p.label for p in EMOTION_PROTOTYPES} | {
    "contempt": "Contempt",
    "neutral": "Neutral",
}
NEGATIVE_EMOTIONS: frozenset[str] = frozenset({"fear", "anger", "disgust", "sadness", "contempt"})
_CONTEMPT_VALENCE = -0.7
_EMOTION_AUS: dict[str, frozenset[str]] = {
    proto.name: frozenset(code for group in proto.core for code in group) | frozenset(proto.support)
    for proto in EMOTION_PROTOTYPES
} | {"contempt": frozenset({"AU12", "AU14"})}


def emotion_involves(emotion: str, codes: Iterable[str]) -> bool:
    """True when any of ``codes`` belongs to the emotion's prototype (core or support)."""
    return not _EMOTION_AUS.get(emotion, frozenset()).isdisjoint(codes)


def _prototype_score(proto: EmotionPrototype, act: Mapping[str, float]) -> float:
    groups = [max(act.get(code, 0.0) for code in group) for group in proto.core]
    if min(groups) <= 0.0:
        return 0.0
    base = math.prod(groups) ** (1.0 / len(groups))
    support = sum(act.get(code, 0.0) for code in proto.support) / max(1, len(proto.support))
    return max(0.0, min(1.0, base * (0.75 + 0.5 * support)))


def contempt_score(left_act: Mapping[str, float], right_act: Mapping[str, float]) -> float:
    """Unilateral lip-corner pull / dimpler (AU12, AU14) — the EMFACS contempt marker."""
    best = 0.0
    for code in ("AU12", "AU14"):
        left = left_act.get(code, 0.0)
        right = right_act.get(code, 0.0)
        strong, weak = max(left, right), min(left, right)
        if strong < 0.2 or weak > 0.45 * strong:
            continue
        best = max(best, strong - weak)
    return max(0.0, min(1.0, best))


def score_emotions(
    act: Mapping[str, float],
    left_act: Mapping[str, float] | None = None,
    right_act: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Prototype match strength per emotion (0-1) from baseline-relative AU activations."""
    scores = {proto.name: _prototype_score(proto, act) for proto in EMOTION_PROTOTYPES}
    scores["contempt"] = contempt_score(left_act or act, right_act or act)
    if scores["contempt"] > 0.0:
        scores["happiness"] *= 1.0 - scores["contempt"]
    return {name: round(value, 3) for name, value in scores.items()}


def dominant_emotion(scores: Mapping[str, float], floor: float = 0.22) -> tuple[str, float]:
    if not scores:
        return "neutral", 0.0
    name, value = max(scores.items(), key=lambda item: item[1])
    if value < floor:
        return "neutral", round(1.0 - value, 3)
    return name, value


def valence(scores: Mapping[str, float]) -> float:
    """Signed affect estimate in [-1, 1] weighted by prototype valence."""
    weights = {proto.name: proto.valence for proto in EMOTION_PROTOTYPES}
    weights["contempt"] = _CONTEMPT_VALENCE
    positive = max((scores.get(n, 0.0) * w for n, w in weights.items() if w > 0), default=0.0)
    negative = max((scores.get(n, 0.0) * -w for n, w in weights.items() if w < 0), default=0.0)
    return round(max(-1.0, min(1.0, positive - negative)), 3)


def activation(act: Mapping[str, float], top_n: int = 3) -> float:
    """Overall facial activation — mean of the strongest AU activations."""
    values = sorted(act.values(), reverse=True)[:top_n]
    return round(sum(values) / top_n, 3) if values else 0.0


@dataclass(frozen=True)
class Cue:
    key: str
    label: str
    meaning: str


CUES: dict[str, Cue] = {
    cue.key: cue
    for cue in (
        Cue("duchenne_smile", "Duchenne smile", "AU6+AU12 — felt-enjoyment marker"),
        Cue("social_smile", "Social smile", "AU12 without AU6 — polite / posed smile"),
        Cue("unilateral_smirk", "Unilateral smirk", "one-sided AU12/AU14 — contempt marker"),
        Cue("lip_compression", "Lip compression", "AU24 — withholding or effortful control"),
        Cue("lip_suck", "Lip suck / bite", "AU28 — tension or self-soothing"),
        Cue("brow_flash", "Eyebrow flash", "brief AU1+AU2 — recognition or emphasis"),
        Cue(
            "speech_emphasis",
            "Speech emphasis",
            "brief brow raise or lower timed with speech — conversational baton, not affect",
        ),
        Cue("distress_brow", "Distress brow", "AU1+AU4 — worry / grief muscles"),
        Cue("focus_frown", "Focused frown", "AU4+AU7 — concentration or annoyance"),
        Cue("nose_wrinkle", "Nose wrinkle", "AU9/AU10 — aversion / disgust leakage"),
        Cue("fear_widen", "Fear widen", "AU5+AU20 — alarm / threat response"),
        Cue("chin_raise", "Chin raise", "AU17 — doubt, defiance, or holding back sadness"),
        Cue("jaw_thrust", "Jaw thrust", "AU29 — defiance or suppressed anger"),
        Cue("cheek_puff", "Cheek puff", "AU34 — exhale, relief, or frustration"),
        Cue("masking", "Masking", "negative movement covered by a smile"),
        Cue("suppression", "Suppression", "negative onset cut off by lip press / chin raise"),
        Cue("leakage", "Leakage", "negative micro-expression during a sustained smile"),
        Cue("blink_burst", "Blink burst", "3+ blinks in quick succession — load / arousal"),
        Cue("eye_closure", "Prolonged eye closure", "eyes closed >400 ms — blocking / fatigue"),
        Cue("gaze_aversion", "Gaze aversion", "sustained look away from camera"),
        Cue("head_nod", "Head nod", "vertical head oscillation — assent / backchannel"),
        Cue("head_shake", "Head shake", "horizontal head oscillation — negation"),
        Cue("self_touch", "Self-touch", "hand-to-face contact — pacifying behaviour"),
    )
}


@dataclass(frozen=True)
class AUEvent:
    code: str
    onset: float
    offset: float
    apex_t: float
    apex_level: float
    side: str
    left: float = 0.0
    right: float = 0.0


NEGATIVE_AU_CODES: frozenset[str] = frozenset({"AU4", "AU9", "AU10", "AU15", "AU20"})
MICRO_MAX_SEC = 0.5


def is_negative_event(event: AUEvent) -> bool:
    return event.code in NEGATIVE_AU_CODES or (event.code == "AU14" and event.side in {"L", "R"})


def episode_cues(
    events: list[AUEvent],
    onset: float,
    duration: float,
    smile_onset: float | None = None,
    press_onset: float | None = None,
) -> list[str]:
    """Cue keys implied by a grouped expression episode.

    ``smile_onset`` / ``press_onset`` are the AU12 and AU24/AU17 onsets in effect
    around the episode (they may belong to a neighbouring episode), which is how
    masking, suppression, and leakage are recognised.
    """
    codes = {e.code for e in events}
    by_code = {e.code: e for e in events}
    cues: list[str] = []
    if "AU12" in codes:
        if by_code["AU12"].side in {"L", "R"} and "AU6" not in codes:
            cues.append("unilateral_smirk")
        elif "AU6" in codes:
            cues.append("duchenne_smile")
        elif duration > 0.5:
            cues.append("social_smile")
    if "AU14" in codes and by_code["AU14"].side in {"L", "R"} and "unilateral_smirk" not in cues:
        cues.append("unilateral_smirk")
    if "AU24" in codes:
        cues.append("lip_compression")
    if "AU28" in codes:
        cues.append("lip_suck")
    if {"AU1", "AU2"} <= codes and duration <= 0.6:
        cues.append("brow_flash")
    if {"AU1", "AU4"} <= codes:
        cues.append("distress_brow")
    if {"AU4", "AU7"} <= codes and duration > 0.5:
        cues.append("focus_frown")
    if codes & {"AU9", "AU10"}:
        cues.append("nose_wrinkle")
    if {"AU5", "AU20"} <= codes:
        cues.append("fear_widen")
    if "AU17" in codes:
        cues.append("chin_raise")
    if "AU29" in codes:
        cues.append("jaw_thrust")
    if "AU34" in codes:
        cues.append("cheek_puff")

    negative = [e for e in events if is_negative_event(e)]
    if negative:
        first_negative = min(e.onset for e in negative)
        if smile_onset is not None:
            if first_negative + 0.1 < smile_onset <= first_negative + 0.8:
                cues.append("masking")
            elif smile_onset < onset - 1.0 and duration <= MICRO_MAX_SEC:
                cues.append("leakage")
        if press_onset is not None and first_negative + 0.05 < press_onset <= first_negative + 0.6:
            cues.append("suppression")
    return cues


def describe_aus(codes: Iterable[str]) -> str:
    return "+".join(sorted(set(codes), key=lambda code: int(code[2:])))
