"""Two-stage serve detection: cheap proposer → short-clip VLM verifier."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pickleball_serve_detection.audio_proposer import (
    audio_proposer_dependencies_available,
    propose_serves_by_gap,
    propose_serves_by_onset,
)
from pickleball_serve_detection.frame_sampler import FrameSampler
from pickleball_serve_detection.serve_detector import ServeDetector


def merge_nearby_timestamps(
    timestamps: list[float],
    min_gap_sec: float,
    *,
    keep: Literal["first", "mean"] = "first",
) -> list[float]:
    """Merge peaks within ``min_gap_sec`` (simple NMS). ``keep``: ``first`` | ``mean``."""
    ts = sorted(float(t) for t in timestamps)
    if not ts:
        return []
    out = [ts[0]]
    for t in ts[1:]:
        if t - out[-1] >= min_gap_sec:
            out.append(t)
        elif keep == "mean":
            out[-1] = (out[-1] + t) / 2.0
    return out


def parse_external_candidates(value: str | Path | None) -> list[float] | None:
    """
    Parse external candidate seconds from a JSON file, or a comma-separated string.

    JSON may be a bare list of numbers or ``{"candidates": [...]}``.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        if not value.is_file():
            return None
        data = json.loads(value.read_text(encoding="utf-8"))
    else:
        s = str(value).strip()
        if not s:
            return None
        p = Path(s)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
        else:
            parts = [x.strip() for x in s.split(",") if x.strip()]
            return [float(x) for x in parts] if parts else None
    if isinstance(data, list):
        return [float(x) for x in data]
    if isinstance(data, dict) and "candidates" in data:
        return [float(x) for x in data["candidates"]]
    raise ValueError(
        "External candidates JSON must be a list of numbers or an object with key 'candidates'."
    )


@dataclass
class CascadeConfig:
    """Configuration for two-stage cascade (proposer → VLM)."""

    use_audio_proposer: bool = True
    audio_mode: Literal["gap", "onset"] = "gap"
    min_gap_before_sec: float = 5.0
    gap_onset_threshold: float = 0.35
    simple_onset_threshold: float = 0.1
    audio_bandpass_hz: tuple[float, float] = (500.0, 6000.0)
    audio_min_distance_sec: float = 5.0
    audio_sr: int = 16000
    audio_hop_length: int = 512
    external_candidates: list[float] | None = None

    window_sec: float = 2.0
    window_offset_sec: float = -1.0
    frames_per_clip: int = 8
    merge_gap_sec: float = 1.5

    extra: dict[str, Any] = field(default_factory=dict)


def extract_frames_for_window(
    video_path: str | Path,
    timestamp_sec: float,
    *,
    window_sec: float = 2.0,
    window_offset_sec: float = -1.0,
    num_frames: int = 8,
    sampler: FrameSampler | None = None,
    prefix: str | None = None,
    jpeg_quality: int = 92,
) -> list[Path]:
    """Extract frames for VLM verification window; returns JPEG paths (chronological)."""
    video_path = Path(video_path)
    fs = sampler or FrameSampler()
    frames = fs.sample_window_with_offset(
        video_path,
        timestamp_sec,
        window_sec=window_sec,
        window_offset_sec=window_offset_sec,
        num_frames=num_frames,
        prefix=prefix,
        jpeg_quality=jpeg_quality,
    )
    return [f.path for f in frames]


def run_cascade(
    video_path: str | Path,
    detector: ServeDetector,
    config: CascadeConfig | None = None,
    *,
    frame_sampler: FrameSampler | None = None,
) -> tuple[list[float], list[dict[str, Any]]]:
    """
    Two-stage serve detection: propose candidates → verify with short-clip VLM → merge YES times.

    Returns ``(verified_timestamps_merged, per_candidate_log)``.
    """
    cfg = config or CascadeConfig()
    video_path = Path(video_path).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(video_path)

    candidates: list[float]
    if cfg.external_candidates is not None and len(cfg.external_candidates) > 0:
        candidates = sorted(float(x) for x in cfg.external_candidates)
    elif cfg.use_audio_proposer:
        if not audio_proposer_dependencies_available():
            raise ImportError(
                "Cascade audio proposer needs librosa/scipy/soundfile. "
                "Install with: pip install pickleball-serve-detection[detection], "
                "or pass external_candidates / disable audio."
            )
        if cfg.audio_mode == "gap":
            lo, hi = cfg.audio_bandpass_hz
            candidates = propose_serves_by_gap(
                video_path,
                sr=cfg.audio_sr,
                hop_length=cfg.audio_hop_length,
                bandpass_low=lo,
                bandpass_high=hi,
                onset_threshold=cfg.gap_onset_threshold,
                min_gap_before_sec=cfg.min_gap_before_sec,
                min_distance_sec=cfg.audio_min_distance_sec,
            )
        else:
            candidates = propose_serves_by_onset(
                video_path,
                threshold=cfg.simple_onset_threshold,
                min_distance_sec=cfg.audio_min_distance_sec,
                sr=cfg.audio_sr,
                hop_length=cfg.audio_hop_length,
            )
    else:
        raise ValueError(
            "Cascade needs proposer candidates: set external_candidates, "
            "or use_audio_proposer=True with audio dependencies installed."
        )

    fs = frame_sampler or FrameSampler()
    log: list[dict[str, Any]] = []
    yes_ts: list[float] = []

    for ts in candidates:
        paths = extract_frames_for_window(
            video_path,
            ts,
            window_sec=cfg.window_sec,
            window_offset_sec=cfg.window_offset_sec,
            num_frames=cfg.frames_per_clip,
            sampler=fs,
            prefix=video_path.stem,
        )
        row: dict[str, Any] = {
            "candidate_timestamp_seconds": ts,
            "n_frames": len(paths),
            "frame_paths": [str(p) for p in paths],
            "is_serve": None,
            "raw_response": None,
            "error": None,
        }
        if not paths:
            row["error"] = "no_frames"
            log.append(row)
            continue
        try:
            res = detector.detect_shortclip_window(paths, anchor_timestamp_seconds=ts)
            row["is_serve"] = res.is_serve
            row["raw_response"] = res.raw_response
            row["confidence"] = res.confidence.value
            row["reasoning"] = res.reasoning
            if res.is_serve:
                yes_ts.append(float(ts))
        except Exception as e:
            row["error"] = str(e)
            warnings.warn(f"Cascade verify failed @ {ts}s: {e}", stacklevel=2)
        log.append(row)

    merged = merge_nearby_timestamps(yes_ts, cfg.merge_gap_sec, keep="first")
    return merged, log
