from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from backend.config import Settings
from backend.micro_expressions import ExpressionAnalyzer, count_swings

FPS = 30.0
_NEUTRAL_SHAPES = (
    "browInnerUp", "browOuterUpLeft", "browOuterUpRight", "browDownLeft", "browDownRight",
    "eyeWideLeft", "eyeWideRight", "cheekSquintLeft", "cheekSquintRight", "eyeSquintLeft",
    "eyeSquintRight", "noseSneerLeft", "noseSneerRight", "mouthUpperUpLeft",
    "mouthUpperUpRight", "mouthSmileLeft", "mouthSmileRight", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight", "mouthPressLeft",
    "mouthPressRight", "mouthStretchLeft", "mouthStretchRight", "mouthShrugLower",
    "eyeBlinkLeft", "eyeBlinkRight", "jawOpen",
)


def _envelope(t: float, start: float, duration: float) -> float:
    """Trapezoid: 60 ms rise, plateau, 60 ms fall."""
    edge = 0.06
    if t < start or t > start + duration:
        return 0.0
    rel = t - start
    return min(1.0, rel / edge, (duration - rel) / edge)


def _synthetic_face(points: np.ndarray, scores: dict[str, float]) -> SimpleNamespace:
    landmarks = [SimpleNamespace(x=float(x), y=float(y), z=0.0) for x, y in points]
    categories = [SimpleNamespace(category_name=k, score=v) for k, v in scores.items()]
    return SimpleNamespace(
        face_landmarks=[landmarks],
        face_blendshapes=[categories],
        facial_transformation_matrixes=[],
    )


def _run_script(analyzer: ExpressionAnalyzer, seconds: float, events, talk_from=None) -> None:
    rng = np.random.default_rng(7)
    base_points = rng.uniform([0.38, 0.3], [0.62, 0.7], size=(478, 2))
    for frame in range(int(seconds * FPS)):
        t = frame / FPS
        scores = {
            name: float(np.clip(0.05 + rng.normal(0, 0.008), 0, 1)) for name in _NEUTRAL_SHAPES
        }
        if talk_from is not None and t >= talk_from:
            scores["jawOpen"] = 0.05 + 0.2 * abs(np.sin(2 * np.pi * 2.5 * t))
        for start, duration, deltas in events:
            level = _envelope(t, start, duration)
            for name, delta in deltas.items():
                scores[name] = float(np.clip(scores.get(name, 0.05) + delta * level, 0, 1))
        points = base_points + rng.normal(0, 0.0004, base_points.shape)
        analyzer.process(_synthetic_face(points, scores), None, t, t)


def test_detects_micro_macro_and_unilateral_episodes() -> None:
    analyzer = ExpressionAnalyzer(Settings(gemini_enabled=False, gemini_api_key=""))
    analyzer.set_enabled(True)
    events = [
        (6.0, 0.27, {"noseSneerLeft": 0.35, "noseSneerRight": 0.35,
                     "mouthUpperUpLeft": 0.3, "mouthUpperUpRight": 0.3}),
        (8.0, 2.0, {"cheekSquintLeft": 0.4, "cheekSquintRight": 0.4,
                    "mouthSmileLeft": 0.55, "mouthSmileRight": 0.55}),
        (12.0, 0.37, {"mouthSmileLeft": 0.35, "mouthDimpleLeft": 0.35}),
        (14.0, 0.15, {"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.8}),
    ]
    _run_script(analyzer, 16.0, events)

    analysis = analyzer.analysis()
    episodes = [e for e in analysis["episodes"] if e["kind"] != "behavior"]
    assert len(episodes) == 3, episodes

    disgust, joy, smirk = episodes
    assert disgust["kind"] == "micro" and disgust["emotion"] == "disgust"
    assert {"AU9", "AU10"} <= set(disgust["aus"])
    assert 5.9 <= disgust["at"] <= 6.2
    assert disgust["notable"] is True

    assert joy["kind"] == "macro" and joy["emotion"] == "happiness"
    assert "duchenne_smile" in joy["cues"]

    assert smirk["kind"] == "micro" and smirk["side"] == "L"
    assert "unilateral_smirk" in smirk["cues"]

    subject = analysis["subjects"][0]
    assert subject["calibrated"] is True
    assert subject["blink"]["total"] == 1


def test_speech_gates_lower_face_and_relabels_brow_batons() -> None:
    analyzer = ExpressionAnalyzer(Settings(gemini_enabled=False, gemini_api_key=""))
    analyzer.set_enabled(True)
    events = [
        (7.0, 0.3, {"mouthPressLeft": 0.4, "mouthPressRight": 0.4}),
        (9.0, 0.25, {"mouthSmileLeft": 0.35, "mouthDimpleLeft": 0.35}),
        (11.0, 0.4, {"browOuterUpLeft": 0.4, "browOuterUpRight": 0.4, "browInnerUp": 0.35}),
        (13.0, 0.3, {"browDownLeft": 0.4, "browDownRight": 0.4}),
        (
            15.0,
            0.3,
            {
                "browDownLeft": 0.45,
                "browDownRight": 0.45,
                "eyeSquintLeft": 0.45,
                "eyeSquintRight": 0.45,
            },
        ),
        (16.0, 0.8, {"mouthPucker": 0.9}),
    ]
    _run_script(analyzer, 18.0, events, talk_from=5.0)
    episodes = analyzer.analysis()["episodes"]
    expressive = [e for e in episodes if e["kind"] != "behavior"]
    assert [(e["aus"], e["emotion"]) for e in expressive] == [(["AU4", "AU7"], "anger")]
    assert expressive[0]["speaking"] is True
    batons = [e for e in episodes if e["cues"] == ["speech_emphasis"]]
    assert [(b["label"], b["aus"]) for b in batons] == [
        ("Speech emphasis · brow raise", ["AU1", "AU2"]),
        ("Speech emphasis · brow lower", ["AU4"]),
    ]
    assert 10.9 <= batons[0]["at"] <= 11.2 and 12.9 <= batons[1]["at"] <= 13.2


def test_episode_emotion_requires_its_own_action_units() -> None:
    analyzer = ExpressionAnalyzer(Settings(gemini_enabled=False, gemini_api_key=""))
    analyzer.set_enabled(True)
    frown = {"browDownLeft": 0.45, "browDownRight": 0.45, "eyeSquintLeft": 0.45,
             "eyeSquintRight": 0.45}
    events = [(6.0, 3.0, frown), (7.5, 0.3, {"mouthPucker": 0.5})]
    _run_script(analyzer, 11.0, events)
    expressive = [e for e in analyzer.analysis()["episodes"] if e["kind"] != "behavior"]
    labels = {tuple(e["aus"]): (e["emotion"], e["label"]) for e in expressive}
    assert labels[("AU4", "AU7")][0] == "anger"
    assert labels[("AU18",)] == (None, "Lip pucker")


def test_neutral_noise_produces_no_episodes() -> None:
    analyzer = ExpressionAnalyzer(Settings(gemini_enabled=False, gemini_api_key=""))
    analyzer.set_enabled(True)
    _run_script(analyzer, 12.0, [])
    analysis = analyzer.analysis()
    assert analysis["counts"]["micro"] == 0
    assert analysis["counts"]["macro"] == 0
    status = analyzer.status()
    assert status["state"] == "live"
    assert status["subjects"][0]["id"] == "A"


def test_disabled_analyzer_reports_off() -> None:
    analyzer = ExpressionAnalyzer(Settings(gemini_enabled=False, gemini_api_key=""))
    assert analyzer.status()["state"] == "off"


def test_count_swings_detects_oscillation() -> None:
    nod = [0, 6, 0, 6, 0, 6, 0]
    assert count_swings(nod, 4.0) >= 4
    assert count_swings([0, 1, 0, 1, 0], 4.0) == 0
