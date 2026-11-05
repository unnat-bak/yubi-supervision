from __future__ import annotations

import numpy as np
from backend.expression_vision import (
    EXPR_LIPS_INDICES,
    EXPR_MESH_CONNECTIONS,
    EXPR_NOSE_INDICES,
    draw_expression_overlay,
)
from backend.micro_expressions import SubjectRender
from mediapipe.tasks.python.vision.face_landmarker import (
    FaceLandmarksConnections as FLC,
)


def test_mesh_includes_lips_and_nose_groups() -> None:
    lip_edges = {(conn.start, conn.end) for conn in FLC.FACE_LANDMARKS_LIPS}
    nose_edges = {(conn.start, conn.end) for conn in FLC.FACE_LANDMARKS_NOSE}
    mesh = set(EXPR_MESH_CONNECTIONS)
    assert lip_edges.issubset(mesh)
    assert nose_edges.issubset(mesh)
    assert len(EXPR_LIPS_INDICES) > 10
    assert len(EXPR_NOSE_INDICES) > 5


def _render(flash: str | None = None) -> SubjectRender:
    rng = np.random.default_rng(3)
    points = rng.uniform(120, 360, size=(478, 2)).astype(np.float32)
    return SubjectRender(
        subject="A",
        points_px=points,
        regions={"brow": 0.6, "eyes": 0.1, "nose": 0.0, "mouth": 0.3},
        tag="A · JOY 64%",
        flash=flash,
        flash_strength=0.8 if flash else 0.0,
        primary=True,
    )


def test_overlay_draws_mesh_tag_and_flash() -> None:
    scene = np.zeros((480, 640, 3), dtype=np.uint8)
    plain = draw_expression_overlay(scene.copy(), [_render()])
    flashed = draw_expression_overlay(scene.copy(), [_render("MICRO · Disgust")])
    assert plain.any()
    assert int(np.abs(flashed.astype(int) - plain.astype(int)).sum()) > 0


def test_overlay_no_renders_is_noop() -> None:
    scene = np.full((120, 160, 3), 7, dtype=np.uint8)
    out = draw_expression_overlay(scene.copy(), [])
    assert np.array_equal(out, scene)
