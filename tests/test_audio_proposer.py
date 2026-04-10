"""Audio proposer tests (optional librosa)."""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from pickleball_serve_detection import audio_proposer


def test_audio_dependencies_flag() -> None:
    assert isinstance(audio_proposer.audio_proposer_dependencies_available(), bool)


@pytest.mark.skipif(
    not audio_proposer.audio_proposer_dependencies_available(),
    reason="librosa/scipy not installed",
)
def test_propose_serves_by_gap_empty_on_silence(tmp_path: Path) -> None:
    wav = tmp_path / "silence.wav"
    _write_silence_wav(wav, duration_sec=0.5, sr=16000)
    out = audio_proposer.propose_serves_by_gap(
        wav,
        min_gap_before_sec=99.0,
        onset_threshold=0.99,
    )
    assert isinstance(out, list)


@pytest.mark.skipif(
    not audio_proposer.audio_proposer_dependencies_available(),
    reason="librosa/scipy not installed",
)
def test_propose_serves_by_onset_returns_sorted(tmp_path: Path) -> None:
    wav = tmp_path / "silence.wav"
    _write_silence_wav(wav, duration_sec=0.3, sr=16000)
    out = audio_proposer.propose_serves_by_onset(wav, threshold=0.99, min_distance_sec=0.05)
    assert out == sorted(out)


def _write_silence_wav(path: Path, *, duration_sec: float, sr: int) -> None:
    n = int(duration_sec * sr)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"\x00\x00" * n)
