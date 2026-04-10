"""Tests for cascade helpers and run_cascade (mocked VLM)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from pickleball_serve_detection.cascade import (
    CascadeConfig,
    merge_nearby_timestamps,
    parse_external_candidates,
    run_cascade,
)
from pickleball_serve_detection.serve_detector import Confidence, ServeDetectionResult


def test_merge_nearby_timestamps_first() -> None:
    assert merge_nearby_timestamps([1.0, 1.4, 5.0], 1.0, keep="first") == [1.0, 5.0]


def test_merge_nearby_timestamps_mean() -> None:
    out = merge_nearby_timestamps([1.0, 2.0], 5.0, keep="mean")
    assert len(out) == 1
    assert abs(out[0] - 1.5) < 1e-6


def test_parse_external_candidates_comma() -> None:
    assert parse_external_candidates(" 1.5 , 3 , 4.0 ") == [1.5, 3.0, 4.0]


def test_parse_external_candidates_json_file(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps([2.0, 4.5]), encoding="utf-8")
    assert parse_external_candidates(str(p)) == [2.0, 4.5]
    assert parse_external_candidates(p) == [2.0, 4.5]


def test_parse_external_candidates_wrapped(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"candidates": [10.0]}), encoding="utf-8")
    assert parse_external_candidates(p) == [10.0]


@patch("pickleball_serve_detection.cascade.extract_frames_for_window")
def test_run_cascade_external_merges_verified(
    mock_extract: MagicMock, tmp_path: Path
) -> None:
    mock_extract.return_value = [tmp_path / "a.jpg"]
    (tmp_path / "a.jpg").write_bytes(b"x")
    video = tmp_path / "v.mp4"
    video.write_bytes(b"fake")

    det = MagicMock()
    det.detect_shortclip_window.return_value = ServeDetectionResult(
        frame_path=tmp_path / "a.jpg",
        is_serve=True,
        confidence=Confidence.HIGH,
        reasoning="yes",
        raw_response="YES",
        timestamp_seconds=1.0,
    )

    ts, log = run_cascade(
        video,
        det,
        CascadeConfig(external_candidates=[1.0, 1.2, 8.0], use_audio_proposer=False),
    )
    assert ts == [1.0, 8.0]
    assert len(log) == 3
    assert all("candidate_timestamp_seconds" in row for row in log)
