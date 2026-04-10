"""Audio-based serve candidate proposer (gap / onset). Requires librosa + scipy."""

from __future__ import annotations

from pathlib import Path

_AUDIO_IMPORT_ERROR: ImportError | None = None
try:
    import librosa  # noqa: F401
    import scipy.signal  # noqa: F401
except ImportError as e:
    _AUDIO_IMPORT_ERROR = e


def audio_proposer_dependencies_available() -> bool:
    return _AUDIO_IMPORT_ERROR is None


def _require_audio() -> None:
    if _AUDIO_IMPORT_ERROR is not None:
        raise ImportError(
            "Audio proposer requires librosa, scipy, and soundfile. "
            "Install with: pip install pickleball-serve-detection[detection]"
        ) from _AUDIO_IMPORT_ERROR


def propose_serves_by_gap(
    video_path: str | Path,
    *,
    sr: int = 16000,
    hop_length: int = 512,
    bandpass_low: float = 1000.0,
    bandpass_high: float = 5000.0,
    onset_threshold: float = 0.35,
    min_gap_before_sec: float = 5.0,
    min_distance_sec: float = 5.0,
) -> list[float]:
    """
    Propose serve timestamps using bandpass-filtered audio onset detection.

    Returns timestamps where a hit follows a long gap (serve preparation time),
    after collapsing dense onset hits. Ported from ``serve_shortclip_finetune_colab.ipynb`` §9b.
    """
    _require_audio()
    import numpy as np
    from scipy.signal import butter, sosfilt

    import librosa

    video_path = Path(video_path)
    y, sr_actual = librosa.load(str(video_path), sr=sr, mono=True)

    nyq = sr_actual / 2.0
    low = max(bandpass_low / nyq, 0.01)
    high = min(bandpass_high / nyq, 0.99)
    if low < high:
        sos = butter(4, [low, high], btype="band", output="sos")
        y_filt = sosfilt(sos, y)
    else:
        y_filt = y

    onset_env = librosa.onset.onset_strength(y=y_filt, sr=sr_actual, hop_length=hop_length)
    mx = float(np.max(onset_env)) if onset_env.size else 0.0
    if mx > 0:
        onset_env = onset_env / mx

    hit_frames = np.where(onset_env >= onset_threshold)[0]
    hit_times = librosa.frames_to_time(hit_frames, sr=sr_actual, hop_length=hop_length)

    collapsed: list[float] = []
    for t in hit_times:
        if not collapsed or t - collapsed[-1] > 0.3:
            collapsed.append(float(t))
    hit_times_arr = np.array(collapsed) if collapsed else np.array([])

    candidates: list[float] = []
    for i, t in enumerate(hit_times_arr):
        gap_before = t - hit_times_arr[i - 1] if i > 0 else t
        if gap_before >= min_gap_before_sec:
            if not candidates or t - candidates[-1] >= min_distance_sec:
                candidates.append(float(t))

    return candidates


def propose_serves_by_onset(
    video_path: str | Path,
    *,
    threshold: float = 0.1,
    min_distance_sec: float = 10.0,
    sr: int = 16000,
    hop_length: int = 512,
) -> list[float]:
    """
    Simple onset peak detection (noisier than gap-based; useful as fallback).

    ``threshold`` is on librosa onset strength after max-normalization.
    """
    _require_audio()
    import numpy as np

    import librosa

    video_path = Path(video_path)
    y, sr_loaded = librosa.load(str(video_path), sr=sr, mono=True)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_loaded, hop_length=hop_length)
    mx = float(np.max(onset_env)) if onset_env.size else 0.0
    if mx > 0:
        onset_env = onset_env / mx

    import scipy.signal

    distance_bins = max(1, int(min_distance_sec * sr_loaded / hop_length))
    peaks, _ = scipy.signal.find_peaks(onset_env, height=threshold, distance=distance_bins)
    timestamps = librosa.frames_to_time(peaks, sr=sr_loaded, hop_length=hop_length)
    return [float(t) for t in timestamps]
