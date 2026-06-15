from __future__ import annotations

import numpy as np
from backend.expression_vision import (
    EXPR_LIPS_INDICES,
    EXPR_MESH_CONNECTIONS,
    EXPR_NOSE_INDICES,
    ExpressionGuidance,
    refine_landmarks_with_guidance,
)
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


def test_refine_landmarks_no_op_without_guidance() -> None:
    points = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
    guidance = ExpressionGuidance()
    refined = refine_landmarks_with_guidance(points, guidance, width=640, height=480)
    assert np.allclose(refined, points)
