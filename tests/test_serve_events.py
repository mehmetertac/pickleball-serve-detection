"""Tests for serve timestamp clustering."""

from types import SimpleNamespace

from pickleball_serve_detection.serve_events import serve_timestamps_from_detections


def test_empty():
    assert serve_timestamps_from_detections([]) == []


def test_merge_cluster_mean():
    hits = [
        SimpleNamespace(is_serve=True, timestamp_seconds=10.0),
        SimpleNamespace(is_serve=True, timestamp_seconds=11.0),
        SimpleNamespace(is_serve=False, timestamp_seconds=12.0),
    ]
    out = serve_timestamps_from_detections(hits, merge_gap_seconds=5.0, cooldown_seconds=1.0)
    assert len(out) == 1
    assert abs(out[0] - 10.5) < 1e-6


def test_cooldown_dedup():
    hits = [
        SimpleNamespace(is_serve=True, timestamp_seconds=0.0),
        SimpleNamespace(is_serve=True, timestamp_seconds=20.0),
    ]
    out = serve_timestamps_from_detections(hits, merge_gap_seconds=1.0, cooldown_seconds=15.0)
    assert out == [0.0, 20.0]

    out2 = serve_timestamps_from_detections(hits, merge_gap_seconds=1.0, cooldown_seconds=25.0)
    assert out2 == [0.0]
