"""Tests for training-style YES/NO serve response parsing."""

from pickleball_serve_detection.serve_detector import Confidence, parse_training_style_serve_response


def test_yes_prefix():
    yes, conf, _ = parse_training_style_serve_response("YES")
    assert yes is True and conf == Confidence.HIGH


def test_yes_with_dash_explanation():
    yes, conf, _ = parse_training_style_serve_response(
        "YES - The player is preparing to serve at the baseline."
    )
    assert yes is True and conf == Confidence.HIGH


def test_no_prefix():
    yes, conf, _ = parse_training_style_serve_response("NO - Players in rally.")
    assert yes is False and conf == Confidence.HIGH


def test_no_override_ball_in_hand():
    yes, conf, _ = parse_training_style_serve_response(
        "NO - The person has the ball in hand at baseline."
    )
    assert yes is True and conf == Confidence.MEDIUM


def test_serve_colon_yes():
    yes, conf, _ = parse_training_style_serve_response("SERVE: YES\nREASON: x")
    assert yes is True and conf == Confidence.HIGH


def test_maybe():
    yes, conf, _ = parse_training_style_serve_response("Maybe — unclear.")
    assert yes is True and conf == Confidence.LOW


def test_empty():
    yes, conf, reason = parse_training_style_serve_response("")
    assert yes is False and conf == Confidence.UNKNOWN and reason == ""
