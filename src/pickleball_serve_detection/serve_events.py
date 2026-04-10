"""Cluster coarse serve-detection samples into serve anchor timestamps."""

from __future__ import annotations

from typing import Protocol


class ServeHitProto(Protocol):
    is_serve: bool
    timestamp_seconds: float | None


def serve_timestamps_from_detections(
    results: list[ServeHitProto],
    *,
    merge_gap_seconds: float = 3.0,
    cooldown_seconds: float = 10.0,
) -> list[float]:
    """
    Build ordered serve anchor times from per-frame VLM (or other) outputs.

    - Merges nearby positive timestamps into one cluster (mean time).
    - Drops anchors closer than ``cooldown_seconds`` (keep earlier).
    """
    positives = sorted(
        r.timestamp_seconds for r in results if r.is_serve and r.timestamp_seconds is not None
    )
    if not positives:
        return []

    clusters: list[list[float]] = []
    cur = [positives[0]]
    for t in positives[1:]:
        if t - cur[-1] <= merge_gap_seconds:
            cur.append(t)
        else:
            clusters.append(cur)
            cur = [t]
    clusters.append(cur)

    anchors = [sum(c) / len(c) for c in clusters]

    filtered: list[float] = []
    for a in sorted(anchors):
        if not filtered or a - filtered[-1] >= cooldown_seconds:
            filtered.append(a)
    return filtered
