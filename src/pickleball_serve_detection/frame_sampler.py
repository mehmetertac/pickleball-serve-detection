"""Smart frame sampling for video analysis."""

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

from pickleball_serve_detection.constants import DEFAULT_DOWNLOAD_DIR


def serve_training_style_window_times(
    end_time_seconds: float,
    *,
    window_before_seconds: float = 2.0,
    num_frames: int = 3,
) -> list[float]:
    """
    Time points for each training-style serve clip: ``[T - window_before, T]`` inclusive.

    Matches ``notebooks/serve_detection_training.ipynb`` (``SERVE_WINDOW_BEFORE`` /
    ``FRAMES_PER_SERVE``): evenly spaced samples ending at the anchor second ``T``.
    """
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    start_t = max(0.0, float(end_time_seconds) - float(window_before_seconds))
    end_t = float(end_time_seconds)
    if num_frames == 1:
        return [end_t]
    step = (end_t - start_t) / (num_frames - 1)
    return [start_t + i * step for i in range(num_frames)]


COARSE_MANIFEST_VERSION = 1


@dataclass
class SampledFrame:
    """A sampled frame with metadata."""

    path: Path
    frame_index: int
    timestamp_seconds: float


def save_coarse_serve_manifest(
    manifest_path: str | Path,
    video_path: str | Path,
    coarse: list[tuple[float, list[SampledFrame]]],
    *,
    interval_seconds: float,
    window_before_seconds: float,
    frames_per_anchor: int,
) -> None:
    """Write coarse VLM window JPEGs layout for a later GPU-only inference step."""
    mp = Path(manifest_path)
    mp.parent.mkdir(parents=True, exist_ok=True)
    data: dict[str, Any] = {
        "version": COARSE_MANIFEST_VERSION,
        "video_file": str(Path(video_path).resolve()),
        "interval_seconds": float(interval_seconds),
        "window_before_seconds": float(window_before_seconds),
        "frames_per_anchor": int(frames_per_anchor),
        "windows": [
            {
                "anchor": float(a),
                "frames": [
                    {
                        "path": str(sf.path.resolve()),
                        "timestamp_seconds": float(sf.timestamp_seconds),
                        "frame_index": int(sf.frame_index),
                    }
                    for sf in seq
                ],
            }
            for a, seq in coarse
        ],
    }
    mp.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_coarse_serve_manifest(
    manifest_path: str | Path,
    *,
    expected_video_path: str | Path,
    interval_seconds: float,
    window_before_seconds: float,
    frames_per_anchor: int,
) -> list[tuple[float, list[SampledFrame]]]:
    """
    Load coarse windows written by :func:`save_coarse_serve_manifest`.

    Raises:
        FileNotFoundError: manifest missing.
        ValueError: video path or sampling params do not match manifest.
    """
    mp = Path(manifest_path)
    if not mp.is_file():
        raise FileNotFoundError(str(mp))
    data = json.loads(mp.read_text(encoding="utf-8"))
    if data.get("version") != COARSE_MANIFEST_VERSION:
        raise ValueError(f"Unsupported coarse manifest version: {data.get('version')!r}")
    exp_v = str(Path(expected_video_path).resolve())
    if data.get("video_file") != exp_v:
        raise ValueError(
            "Manifest video_file does not match current video (re-run coarse CPU step for this file): "
            f"manifest={data.get('video_file')!r} vs current={exp_v!r}"
        )
    checks = [
        ("interval_seconds", data.get("interval_seconds"), float(interval_seconds)),
        ("window_before_seconds", data.get("window_before_seconds"), float(window_before_seconds)),
        ("frames_per_anchor", data.get("frames_per_anchor"), int(frames_per_anchor)),
    ]
    for name, got, want in checks:
        if got != want:
            raise ValueError(
                f"Manifest {name}={got!r} does not match config {want!r} — re-run coarse CPU step or fix config."
            )
    out: list[tuple[float, list[SampledFrame]]] = []
    for w in data.get("windows", []):
        seq = [
            SampledFrame(
                Path(f["path"]),
                int(f["frame_index"]),
                float(f["timestamp_seconds"]),
            )
            for f in w["frames"]
        ]
        out.append((float(w["anchor"]), seq))
    return out


class FrameSampler:
    """
    Smart frame sampler for video analysis.

    Provides various sampling strategies optimized for serve detection
    and other video analysis tasks.
    """

    def __init__(self, output_dir: str | Path | None = None):
        """
        Initialize the frame sampler.

        Args:
            output_dir: Directory to save sampled frames.
        """
        if output_dir is None:
            output_dir = DEFAULT_DOWNLOAD_DIR / "sampled_frames"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sample_uniform(
        self,
        video_path: str | Path,
        interval_seconds: float = 0.5,
        start_time: float = 0.0,
        end_time: float | None = None,
        prefix: str | None = None,
    ) -> list[SampledFrame]:
        """
        Sample frames at uniform time intervals.

        Args:
            video_path: Path to the video file.
            interval_seconds: Time between samples in seconds.
            start_time: Start time in seconds.
            end_time: End time in seconds (None = end of video).
            prefix: Prefix for output filenames.

        Returns:
            List of SampledFrame objects.
        """
        video_path = Path(video_path)
        if prefix is None:
            prefix = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_time is None or end_time > duration:
            end_time = duration

        sampled_frames = []
        current_time = start_time

        while current_time < end_time:
            frame_idx = int(current_time * fps)

            if frame_idx >= total_frames:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                output_path = self.output_dir / f"{prefix}_t{current_time:.2f}s.jpg"
                cv2.imwrite(str(output_path), frame)
                sampled_frames.append(
                    SampledFrame(
                        path=output_path,
                        frame_index=frame_idx,
                        timestamp_seconds=current_time,
                    )
                )

            current_time += interval_seconds

        cap.release()
        return sampled_frames

    def sample_coarse_serve_training_windows(
        self,
        video_path: str | Path,
        *,
        interval_seconds: float = 1.0,
        window_before_seconds: float = 2.0,
        frames_per_anchor: int = 3,
        start_time: float = 0.0,
        end_time: float | None = None,
        prefix: str | None = None,
        progress_every_n_anchors: int = 0,
    ) -> list[tuple[float, list[SampledFrame]]]:
        """
        Coarse VLM scan: every ``interval_seconds``, extract a training-style window
        ending at that anchor time (same geometry as serve_detection_training.ipynb).

        Args:
            progress_every_n_anchors: If > 0, ``print`` status every N anchors (Colab/Drive I/O
                can make this step take many minutes with no GPU use).

        Returns:
            List of ``(anchor_time_seconds, frames_chronological)`` per anchor.
        """
        if frames_per_anchor < 1:
            raise ValueError("frames_per_anchor must be >= 1")

        video_path = Path(video_path)
        if prefix is None:
            prefix = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        if end_time is None or end_time > duration:
            end_time = duration

        est_anchors = max(0, int((end_time - start_time) / interval_seconds))
        if progress_every_n_anchors > 0:
            print(
                f"Coarse frame extract: ~{est_anchors} anchors × {frames_per_anchor} frames "
                f"(CPU/Disk; Google Drive can be very slow). Showing progress every "
                f"{progress_every_n_anchors} anchors..."
            )
        t0 = time.perf_counter()

        out: list[tuple[float, list[SampledFrame]]] = []
        anchor = float(start_time)
        anchor_idx = 0

        while anchor < end_time:
            time_points = serve_training_style_window_times(
                anchor,
                window_before_seconds=window_before_seconds,
                num_frames=frames_per_anchor,
            )
            seq: list[SampledFrame] = []
            for wi, tsec in enumerate(time_points):
                frame_idx = int(tsec * fps)
                if frame_idx >= total_frames:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    fname = f"{prefix}_a{anchor_idx:05d}_{anchor:.2f}s_w{wi:02d}_t{tsec:.2f}s.jpg"
                    output_path = self.output_dir / fname
                    cv2.imwrite(str(output_path), frame)
                    seq.append(
                        SampledFrame(
                            path=output_path,
                            frame_index=frame_idx,
                            timestamp_seconds=tsec,
                        )
                    )
            if seq:
                out.append((anchor, seq))
            anchor += interval_seconds
            anchor_idx += 1
            if progress_every_n_anchors > 0 and anchor_idx % progress_every_n_anchors == 0:
                elapsed = time.perf_counter() - t0
                print(
                    f"  … anchor #{anchor_idx} @ {anchor:.1f}s / {end_time:.0f}s "
                    f"({elapsed:.0f}s elapsed)"
                )

        cap.release()
        if progress_every_n_anchors > 0:
            print(
                f"Coarse frame extract done: {len(out)} windows in "
                f"{time.perf_counter() - t0:.0f}s."
            )
        return out

    def sample_window(
        self,
        video_path: str | Path,
        center_time: float,
        window_seconds: float = 2.0,
        num_frames: int = 5,
        prefix: str | None = None,
    ) -> list[SampledFrame]:
        """
        Sample frames around a specific time point.

        Useful for analyzing temporal context around a potential event.

        Args:
            video_path: Path to the video file.
            center_time: Center time of the window in seconds.
            window_seconds: Total window duration in seconds.
            num_frames: Number of frames to extract.
            prefix: Prefix for output filenames.

        Returns:
            List of SampledFrame objects.
        """
        video_path = Path(video_path)
        if prefix is None:
            prefix = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Calculate start and end times
        half_window = window_seconds / 2
        start_time = max(0, center_time - half_window)
        end_time = min(duration, center_time + half_window)

        # Calculate time points
        if num_frames == 1:
            time_points = [center_time]
        else:
            step = (end_time - start_time) / (num_frames - 1)
            time_points = [start_time + i * step for i in range(num_frames)]

        sampled_frames = []

        for i, t in enumerate(time_points):
            frame_idx = int(t * fps)

            if frame_idx >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                output_path = self.output_dir / f"{prefix}_window_{i:02d}_t{t:.2f}s.jpg"
                cv2.imwrite(str(output_path), frame)
                sampled_frames.append(
                    SampledFrame(
                        path=output_path,
                        frame_index=frame_idx,
                        timestamp_seconds=t,
                    )
                )

        cap.release()
        return sampled_frames

    def sample_window_with_offset(
        self,
        video_path: str | Path,
        timestamp_sec: float,
        *,
        window_sec: float = 2.0,
        window_offset_sec: float = 0.0,
        num_frames: int = 8,
        prefix: str | None = None,
        jpeg_quality: int = 92,
    ) -> list[SampledFrame]:
        """
        Sample frames over ``[anchor - window_sec, anchor]`` where
        ``anchor = timestamp_sec + window_offset_sec``.

        Matches the cascade window in ``serve_shortclip_finetune_colab.ipynb`` §9b
        (frame-index linspace between start/end). Use a negative ``window_offset_sec``
        when the proposer fires AFTER ball contact (typical for audio onset detection,
        which is delayed ~0.5–1.5s) so the window shifts backward to cover the strike.
        """
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")

        video_path = Path(video_path)
        if prefix is None:
            prefix = video_path.stem

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 1e-3:
            fps = 30.0

        anchor = float(timestamp_sec) + float(window_offset_sec)
        start_t = max(0.0, anchor - float(window_sec))
        end_t = anchor
        start_frame = int(start_t * fps)
        end_frame = int(end_t * fps)
        if n_total > 0:
            end_frame = min(end_frame, n_total - 1)
            start_frame = min(start_frame, end_frame)

        if num_frames == 1:
            frame_idxs = [end_frame]
        else:
            span = max(0, end_frame - start_frame)
            raw = [start_frame + (span * i / (num_frames - 1)) for i in range(num_frames)]
            frame_idxs = sorted({int(round(x)) for x in raw})
            frame_idxs = [min(max(0, fi), end_frame) for fi in frame_idxs]

        stem = f"{float(timestamp_sec):.3f}".replace(".", "p")
        sampled: list[SampledFrame] = []
        for i, f_idx in enumerate(frame_idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f_idx))
            ret, frame = cap.read()
            if not ret:
                continue
            tsec = f_idx / fps if fps > 0 else 0.0
            output_path = self.output_dir / f"{prefix}_cascade_{stem}_f{i:03d}_t{tsec:.2f}s.jpg"
            cv2.imwrite(
                str(output_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            sampled.append(
                SampledFrame(
                    path=output_path,
                    frame_index=int(f_idx),
                    timestamp_seconds=float(tsec),
                )
            )

        cap.release()
        return sampled

    def sample_for_serve_detection(
        self,
        video_path: str | Path,
        interval_seconds: float = 1.0,
        context_frames: int = 3,
        context_window_seconds: float = 1.5,
        prefix: str | None = None,
    ) -> list[list[SampledFrame]]:
        """
        Sample frames optimized for serve detection.

        This method samples candidate frames at regular intervals,
        then extracts context frames around each candidate for
        temporal analysis.

        Args:
            video_path: Path to the video file.
            interval_seconds: Interval between candidate frames.
            context_frames: Number of context frames per candidate.
            context_window_seconds: Duration of context window.
            prefix: Prefix for output filenames.

        Returns:
            List of frame sequences (each sequence is a list of SampledFrame).
        """
        video_path = Path(video_path)
        if prefix is None:
            prefix = video_path.stem

        # First, get candidate timestamps
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        cap.release()

        # Generate candidate timestamps
        candidate_times = []
        current_time = context_window_seconds / 2  # Start after half window
        while current_time < duration - context_window_seconds / 2:
            candidate_times.append(current_time)
            current_time += interval_seconds

        # Sample windows around each candidate
        sequences = []
        for i, center_time in enumerate(candidate_times):
            sequence = self.sample_window(
                video_path=video_path,
                center_time=center_time,
                window_seconds=context_window_seconds,
                num_frames=context_frames,
                prefix=f"{prefix}_candidate_{i:04d}",
            )
            if sequence:
                sequences.append(sequence)

        return sequences

    def get_video_info(self, video_path: str | Path) -> dict:
        """
        Get video metadata.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing video metadata.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_seconds": duration,
        }


def sample_window_with_offset(
    video_path: str | Path,
    timestamp_sec: float,
    *,
    window_sec: float = 2.0,
    window_offset_sec: float = 0.0,
    num_frames: int = 8,
    output_dir: Path | None = None,
    jpeg_quality: int = 92,
    prefix: str | None = None,
) -> list[SampledFrame]:
    """Module-level helper; see :meth:`FrameSampler.sample_window_with_offset`."""
    return FrameSampler(output_dir=output_dir).sample_window_with_offset(
        video_path,
        timestamp_sec,
        window_sec=window_sec,
        window_offset_sec=window_offset_sec,
        num_frames=num_frames,
        prefix=prefix,
        jpeg_quality=jpeg_quality,
    )
