"""Aggregate per-frame serve votes for one coarse anchor time."""

from __future__ import annotations

from pathlib import Path

from pickleball_serve_detection.serve_detector import Confidence, ServeDetectionResult


def _aggregate_coarse_window_detects(
    anchor: float,
    paths: list[Path],
    per_frame: list[ServeDetectionResult],
    *,
    mode: str,
) -> ServeDetectionResult:
    """Combine single-image ``detect`` results for one coarse anchor (matches LoRA training)."""
    n = len(per_frame)
    if n == 0:
        return ServeDetectionResult(
            frame_path=Path("."),
            is_serve=False,
            confidence=Confidence.UNKNOWN,
            reasoning="No frames in window",
            timestamp_seconds=anchor,
        )
    n_yes = sum(1 for r in per_frame if r.is_serve)
    if mode == "per_frame_any":
        is_serve = n_yes > 0
    elif mode == "per_frame_majority":
        is_serve = n_yes * 2 > n
    else:
        raise ValueError(f"Invalid aggregate mode for per-frame path: {mode}")

    winners = [r for r in per_frame if r.is_serve == is_serve]
    _rank = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1, Confidence.UNKNOWN: 0}
    best = (
        max(winners, key=lambda r: _rank[r.confidence]).confidence
        if winners
        else Confidence.UNKNOWN
    )

    lines = [
        f"[{i+1}/{n}] t={r.timestamp_seconds:.2f}s serve={r.is_serve}: {r.reasoning[:120]}"
        for i, r in enumerate(per_frame)
    ]
    reasoning = f"aggregate={mode} votes_yes={n_yes}/{n}. " + " | ".join(lines)

    return ServeDetectionResult(
        frame_path=paths[-1],
        is_serve=is_serve,
        confidence=best,
        reasoning=reasoning,
        raw_response="\n---\n".join(r.raw_response for r in per_frame if r.raw_response),
        timestamp_seconds=anchor,
    )


def aggregate_coarse_window_serve_votes(
    anchor: float,
    paths: list[Path],
    per_frame: list[ServeDetectionResult],
    *,
    mode: str,
) -> ServeDetectionResult:
    """Combine per-frame ``detect`` outputs for one coarse anchor (notebook / tools)."""
    return _aggregate_coarse_window_detects(anchor, paths, per_frame, mode=mode)
