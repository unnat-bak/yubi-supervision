from __future__ import annotations

import time

import numpy as np
import supervision as sv
from backend.identity_vision import (
    IdentityCache,
    apply_identity_labels,
    collect_person_hints,
    is_person_label,
)


def _person_detection(
    tracker_id: int, xyxy: tuple[int, int, int, int]
) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([xyxy], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        tracker_id=np.array([tracker_id], dtype=np.int32),
    )


def test_is_person_label() -> None:
    assert is_person_label("person")
    assert is_person_label("Player")
    assert not is_person_label("chair")


def test_collect_person_hints_filters_non_people() -> None:
    detections = _person_detection(7, (100, 50, 200, 400))
    tracks = [{"label": "person", "confidence": 0.9, "tracker_id": 7}]
    hints = collect_person_hints(tracks, detections, 1000, 1000, max_persons=6)
    assert len(hints) == 1
    assert hints[0]["tracker_id"] == 7
    assert len(hints[0]["box_2d"]) == 4

    chair_tracks = [{"label": "chair", "confidence": 0.9, "tracker_id": 7}]
    assert collect_person_hints(chair_tracks, detections, 1000, 1000, 6) == []


def test_collect_person_hints_respects_cap() -> None:
    boxes = sv.Detections(
        xyxy=np.array([[i, i, i + 10, i + 50] for i in range(10)], dtype=np.float32),
        confidence=np.full(10, 0.9, dtype=np.float32),
        class_id=np.zeros(10, dtype=np.int32),
        tracker_id=np.arange(10, dtype=np.int32),
    )
    tracks = [
        {"label": "person", "confidence": 0.9, "tracker_id": i} for i in range(10)
    ]
    hints = collect_person_hints(tracks, boxes, 1000, 1000, max_persons=3)
    assert len(hints) == 3


def test_identity_cache_ttl_and_prune() -> None:
    cache = IdentityCache(max_age_sec=0.05)
    cache.set(3, "#23 Robinson", 0.8)
    assert cache.get(3).identity == "#23 Robinson"
    assert cache.fresh_ids() == {3}

    time.sleep(0.06)
    assert cache.get(3) is None  # expired

    cache.set(4, "#0 Tatum", 0.9)
    cache.prune(active_ids={9})
    assert cache.get(4) is None  # pruned (not active)


def test_apply_identity_labels_prefixes() -> None:
    detections = _person_detection(7, (100, 50, 200, 400))
    cache = IdentityCache(max_age_sec=60)
    cache.set(7, "#23 Robinson", 0.8)
    labels = ["person #7 90%"]
    out = apply_identity_labels(detections, labels, cache)
    assert out == ["#23 Robinson · person #7 90%"]


def test_apply_identity_labels_noop_without_entry() -> None:
    detections = _person_detection(7, (100, 50, 200, 400))
    cache = IdentityCache(max_age_sec=60)
    labels = ["person #7 90%"]
    assert apply_identity_labels(detections, labels, cache) == labels
