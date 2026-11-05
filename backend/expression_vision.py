"""Expression overlay — smoothed facial mesh, region heat, subject tags, micro flashes.

Analysis lives in ``backend.micro_expressions``; this module only draws the
``SubjectRender`` output so the overlay always matches what the engine measured.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarksConnections as FLC,
)

from backend.micro_expressions import SubjectRender

# Expression mesh edges — structural groups only (no full tessellation; too heavy per frame).
EXPR_MESH_CONNECTIONS: tuple[tuple[int, int], ...] = tuple(
    (conn.start, conn.end)
    for group in (
        FLC.FACE_LANDMARKS_FACE_OVAL,
        FLC.FACE_LANDMARKS_LEFT_EYEBROW,
        FLC.FACE_LANDMARKS_RIGHT_EYEBROW,
        FLC.FACE_LANDMARKS_LEFT_EYE,
        FLC.FACE_LANDMARKS_RIGHT_EYE,
        FLC.FACE_LANDMARKS_LEFT_IRIS,
        FLC.FACE_LANDMARKS_RIGHT_IRIS,
        FLC.FACE_LANDMARKS_NOSE,
        FLC.FACE_LANDMARKS_LIPS,
    )
    for conn in group
)


def _connection_indices(group: Any) -> frozenset[int]:
    return frozenset({conn.start for conn in group} | {conn.end for conn in group})


EXPR_BROW_INDICES = _connection_indices(FLC.FACE_LANDMARKS_LEFT_EYEBROW) | _connection_indices(
    FLC.FACE_LANDMARKS_RIGHT_EYEBROW
)
EXPR_EYE_INDICES = _connection_indices(FLC.FACE_LANDMARKS_LEFT_EYE) | _connection_indices(
    FLC.FACE_LANDMARKS_RIGHT_EYE
)
EXPR_IRIS_INDICES = _connection_indices(FLC.FACE_LANDMARKS_LEFT_IRIS) | _connection_indices(
    FLC.FACE_LANDMARKS_RIGHT_IRIS
)
EXPR_NOSE_INDICES = _connection_indices(FLC.FACE_LANDMARKS_NOSE)
EXPR_LIPS_INDICES = _connection_indices(FLC.FACE_LANDMARKS_LIPS)

_REGION_INDICES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("brow", tuple(sorted(EXPR_BROW_INDICES))),
    ("eyes", tuple(sorted(EXPR_EYE_INDICES | EXPR_IRIS_INDICES))),
    ("nose", tuple(sorted(EXPR_NOSE_INDICES))),
    ("mouth", tuple(sorted(EXPR_LIPS_INDICES))),
)

# BGR — pearl base, sage-teal signal (matches --signal), warm alert (matches --alert).
_MESH = (132, 144, 158)
_DOT_BASE = np.array((210, 204, 197), dtype=np.float32)
_DOT_ACTIVE = np.array((176, 184, 142), dtype=np.float32)
_FLASH = (106, 168, 201)
_TAG_PLATE = (24, 20, 16)
_TAG_TEXT = (226, 230, 232)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _blend(level: float) -> tuple[int, int, int]:
    mixed = _DOT_BASE + (_DOT_ACTIVE - _DOT_BASE) * max(0.0, min(1.0, level * 1.6))
    return int(mixed[0]), int(mixed[1]), int(mixed[2])


def _draw_tag(
    scene: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.42,
) -> int:
    # Hershey fonts only cover ASCII; anything else renders as "??".
    text = text.replace("·", "|").replace("—", "-").encode("ascii", "ignore").decode()
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, scale, 1)
    x = max(0, min(scene.shape[1] - text_w - 10, anchor[0]))
    y = max(text_h + baseline + 4, anchor[1])
    cv2.rectangle(scene, (x, y - text_h - baseline - 6), (x + text_w + 10, y), _TAG_PLATE, -1)
    cv2.putText(scene, text, (x + 5, y - baseline - 2), _FONT, scale, color, 1, cv2.LINE_AA)
    return y - text_h - baseline - 6


def draw_expression_overlay(scene: np.ndarray, renders: list[SubjectRender]) -> np.ndarray:
    for render in renders:
        points = render.points_px
        if len(points) < 468:
            continue
        pts = points.astype(np.int32)
        for start, end in EXPR_MESH_CONNECTIONS:
            if start < len(pts) and end < len(pts):
                cv2.line(scene, tuple(pts[start]), tuple(pts[end]), _MESH, 1, cv2.LINE_AA)

        for region, indices in _REGION_INDICES:
            level = render.regions.get(region, 0.0)
            color = _blend(level)
            radius = 2 if level >= 0.3 else 1
            for idx in indices:
                if idx < len(pts):
                    cv2.circle(scene, tuple(pts[idx]), radius, color, -1, cv2.LINE_AA)

        x1, y1 = (int(v) for v in pts.min(axis=0))
        x2, y2 = (int(v) for v in pts.max(axis=0))
        top = _draw_tag(scene, render.tag, (x1, y1 - 8), _TAG_TEXT)
        if render.flash:
            _draw_tag(scene, render.flash, (x1, top - 4), _FLASH, scale=0.46)
            thickness = 2 if render.flash_strength > 0.5 else 1
            corner = max(10, int((x2 - x1) * 0.12))
            for cx, cy, dx, dy in (
                (x1, y1, 1, 1),
                (x2, y1, -1, 1),
                (x1, y2, 1, -1),
                (x2, y2, -1, -1),
            ):
                cv2.line(scene, (cx, cy), (cx + dx * corner, cy), _FLASH, thickness, cv2.LINE_AA)
                cv2.line(scene, (cx, cy), (cx, cy + dy * corner), _FLASH, thickness, cv2.LINE_AA)
    return scene
