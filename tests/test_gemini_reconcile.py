from __future__ import annotations

import time

import numpy as np
import supervision as sv

from backend.gemini_vision import (
    GeminiInsight,
    GeminiObject,
    TrackLabelCache,
    labels_match,
    reconcile_tracked_objects,
)


def _detections_with_track(
    tracker_id: int, class_id: int, confidence: float, xyxy: tuple[int, int, int, int]
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([confidence], dtype=np.float32),
        class_id=np.array([class_id], dtype=np.int32),
        tracker_id=np.array([tracker_id], dtype=np.int32),
    )


def test_labels_match_dining_table_and_shelf() -> None:
    assert labels_match("shelf", "dining table")


def test_reconcile_suppresses_unverified_low_confidence() -> None:
    cache = TrackLabelCache(max_age_sec=30.0)
    tracks = [
        {"label": "toothbrush", "confidence": 0.18, "tracker_id": 3},
        {"label": "person", "confidence": 0.92, "tracker_id": 1},
    ]
    detections = _detections_with_track(3, 79, 0.18, (100, 100, 140, 140))
    insight = GeminiInsight(state="ready", objects=[], updated_at=time.time())

    reconciled = reconcile_tracked_objects(
        tracks,
        detections,
        insight,
        width=640,
        height=480,
        cache=cache,
        verify_below=0.45,
        insight_max_age_sec=6.0,
        min_display_conf=0.15,
    )

    labels = {t["label"] for t in reconciled}
    assert "toothbrush" not in labels
    assert "person" in labels


def test_reconcile_corrects_label_from_v3_and_caches() -> None:
    cache = TrackLabelCache(max_age_sec=30.0)
    tracks = [{"label": "dining table", "confidence": 0.22, "tracker_id": 6}]
    detections = _detections_with_track(6, 60, 0.22, (400, 80, 520, 220))
    insight = GeminiInsight(
        state="ready",
        objects=[
            GeminiObject(
                label="shelf",
                confidence=0.78,
                box_2d=[120, 620, 460, 820],
            )
        ],
        updated_at=time.time(),
    )

    reconciled = reconcile_tracked_objects(
        tracks,
        detections,
        insight,
        width=640,
        height=480,
        cache=cache,
        verify_below=0.45,
        insight_max_age_sec=6.0,
        min_display_conf=0.15,
    )

    assert reconciled[0]["label"] == "shelf"
    assert reconciled[0]["verified"] is True

    # Sticky cache: next frame without fresh insight still uses v3 label.
    stale_insight = GeminiInsight(state="ready", objects=[], updated_at=0.0)
    reconciled_again = reconcile_tracked_objects(
        tracks,
        detections,
        stale_insight,
        width=640,
        height=480,
        cache=cache,
        verify_below=0.45,
        insight_max_age_sec=6.0,
        min_display_conf=0.15,
    )
    assert reconciled_again[0]["label"] == "shelf"
    assert reconciled_again[0]["source"] == "v3"
