"""Tests for coarse serve-window vote aggregation."""

from pathlib import Path

from pickleball_serve_detection.coarse_window import aggregate_coarse_window_serve_votes
from pickleball_serve_detection.serve_detector import Confidence, ServeDetectionResult


def _r(path: str, yes: bool, t: float) -> ServeDetectionResult:
    return ServeDetectionResult(
        Path(path),
        is_serve=yes,
        confidence=Confidence.HIGH,
        reasoning="x",
        timestamp_seconds=t,
    )


def test_per_frame_majority_two_of_three():
    per = [_r("a", True, 1.0), _r("b", False, 2.0), _r("c", True, 3.0)]
    paths = [r.frame_path for r in per]
    out = aggregate_coarse_window_serve_votes(10.0, paths, per, mode="per_frame_majority")
    assert out.is_serve is True
    assert out.timestamp_seconds == 10.0


def test_per_frame_majority_two_way_tie_false():
    per = [_r("a", True, 1.0), _r("b", False, 2.0)]
    paths = [r.frame_path for r in per]
    out = aggregate_coarse_window_serve_votes(10.0, paths, per, mode="per_frame_majority")
    assert out.is_serve is False


def test_per_frame_any():
    per = [_r("a", False, 1.0), _r("b", True, 2.0)]
    paths = [r.frame_path for r in per]
    out = aggregate_coarse_window_serve_votes(10.0, paths, per, mode="per_frame_any")
    assert out.is_serve is True
