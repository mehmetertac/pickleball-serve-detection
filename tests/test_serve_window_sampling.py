"""Tests for training-style serve window time sampling."""

from pathlib import Path

from pickleball_serve_detection.frame_sampler import (
    SampledFrame,
    load_coarse_serve_manifest,
    save_coarse_serve_manifest,
    serve_training_style_window_times,
)


def test_serve_training_style_window_times_three_frames_two_second_window():
    times = serve_training_style_window_times(
        10.0, window_before_seconds=2.0, num_frames=3
    )
    assert times == [8.0, 9.0, 10.0]


def test_serve_training_style_window_times_clamps_start_at_zero():
    times = serve_training_style_window_times(
        1.0, window_before_seconds=2.0, num_frames=3
    )
    assert times == [0.0, 0.5, 1.0]


def test_serve_training_style_window_times_single_frame_is_end():
    times = serve_training_style_window_times(
        5.5, window_before_seconds=2.0, num_frames=1
    )
    assert times == [5.5]


def test_coarse_manifest_roundtrip(tmp_path):
    vid = tmp_path / "fake.mp4"
    vid.write_bytes(b"x")
    coarse = [
        (
            2.0,
            [
                SampledFrame(tmp_path / "a.jpg", 10, 1.0),
                SampledFrame(tmp_path / "b.jpg", 20, 2.0),
            ],
        )
    ]
    for sf in coarse[0][1]:
        sf.path.write_bytes(b"jpeg")
    man = tmp_path / "m.json"
    save_coarse_serve_manifest(
        man,
        vid,
        coarse,
        interval_seconds=1.0,
        window_before_seconds=2.0,
        frames_per_anchor=2,
    )
    back = load_coarse_serve_manifest(
        man,
        expected_video_path=vid,
        interval_seconds=1.0,
        window_before_seconds=2.0,
        frames_per_anchor=2,
    )
    assert len(back) == 1
    assert back[0][0] == 2.0
    assert [f.timestamp_seconds for f in back[0][1]] == [1.0, 2.0]
    assert all(isinstance(f.path, Path) for f in back[0][1])
