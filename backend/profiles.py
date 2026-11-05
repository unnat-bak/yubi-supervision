"""Analysis profiles — layer presets and session behavior per use case."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AnalysisProfile:
    key: str
    label: str
    description: str
    layers: dict[str, bool] = field(default_factory=dict)
    # None keeps SOURCE_LOOP; False plays files once so the session ends at EOF.
    loop_files: bool | None = None
    insights: bool = True
    remote_live_lite: bool = True

    def as_dict(self) -> dict[str, object]:
        return {"key": self.key, "label": self.label, "description": self.description}


PROFILES: dict[str, AnalysisProfile] = {
    profile.key: profile
    for profile in (
        AnalysisProfile(
            key="general",
            label="General",
            description="Objects, skeleton, face, hands, and scene intelligence.",
            layers={
                "show_objects": True,
                "show_pose": True,
                "show_face": True,
                "show_hands": True,
                "show_gemini": True,
                "show_expressions": False,
            },
        ),
        AnalysisProfile(
            key="interview",
            label="Interview",
            description=(
                "Micro-expressions, FACS action units, blink / gaze / pulse, and body "
                "language. Files play once and end with a behavioral report."
            ),
            layers={
                "show_objects": False,
                "show_pose": False,
                "show_face": False,
                "show_hands": True,
                "show_gemini": True,
                "show_expressions": True,
                "show_masks": False,
                "show_pose_labels": False,
                "show_identity": False,
            },
            loop_files=False,
            remote_live_lite=False,
        ),
        AnalysisProfile(
            key="security",
            label="Security",
            description="Object tracking, identity labels, and watchlist alerts.",
            layers={
                "show_objects": True,
                "show_pose": False,
                "show_face": False,
                "show_hands": False,
                "show_gemini": True,
                "show_expressions": False,
                "show_identity": True,
            },
            insights=False,
        ),
    )
}


def resolve_profile(key: str | None, default: str = "general") -> AnalysisProfile:
    for candidate in (key, default, "general"):
        if candidate and candidate.strip().lower() in PROFILES:
            return PROFILES[candidate.strip().lower()]
    return PROFILES["general"]
