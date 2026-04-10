"""Serve detection module for pickleball videos."""

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from pickleball_serve_detection.prompts import (
    SERVE_QUICK_PROMPT,
    get_sequence_prompt,
    get_serve_detection_prompt,
    get_serve_training_style_prompt,
    get_shortclip_training_style_prompt,
)
from pickleball_serve_detection.vlm_client import VLMClient

PromptStyle = Literal["training", "structured"]


class Confidence(Enum):
    """Confidence level for detections."""

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class ServeDetectionResult:
    """Result of serve detection on a single frame."""

    frame_path: Path
    is_serve: bool
    confidence: Confidence
    reasoning: str
    player_position: str | None = None
    raw_response: str = ""
    timestamp_seconds: float | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "frame_path": str(self.frame_path),
            "is_serve": self.is_serve,
            "confidence": self.confidence.value,
            "reasoning": self.reasoning,
            "player_position": self.player_position,
            "timestamp_seconds": self.timestamp_seconds,
        }


@dataclass
class ServeSequenceResult:
    """Result of serve detection on a sequence of frames."""

    frame_paths: list[Path]
    is_serve: bool
    confidence: Confidence
    serve_frame_index: int | None
    reasoning: str
    raw_response: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "frame_paths": [str(p) for p in self.frame_paths],
            "is_serve": self.is_serve,
            "confidence": self.confidence.value,
            "serve_frame_index": self.serve_frame_index,
            "reasoning": self.reasoning,
        }


class ServeDetector:
    """Detect serves in pickleball video frames using Vision-Language Models."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        vlm_client: VLMClient | None = None,
        load_in_4bit: bool = True,
        lora_adapter_path: str | Path | None = None,
    ):
        """
        Initialize the serve detector.

        Args:
            model_name: HuggingFace model name for the VLM.
            vlm_client: Optional pre-configured VLM client.
            load_in_4bit: Whether to use 4-bit quantization.
            lora_adapter_path: Optional directory with PEFT LoRA weights for serve detection.
        """
        if vlm_client is not None:
            self._client = vlm_client
        else:
            self._client = VLMClient(
                model_name=model_name,
                load_in_4bit=load_in_4bit,
                adapter_path=lora_adapter_path,
            )

    def load_model(self) -> None:
        """Pre-load the model into memory."""
        self._client.load_model()

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        self._client.unload_model()

    def _default_prompt_style(self) -> PromptStyle:
        """LoRA checkpoints from ``serve_detection_training.ipynb`` expect training-style YES/NO."""
        return "training" if self._client.adapter_path is not None else "structured"

    def _resolve_prompt_style(self, prompt_style: PromptStyle | None) -> PromptStyle:
        return prompt_style if prompt_style is not None else self._default_prompt_style()

    def detect(
        self,
        frame_path: str | Path,
        quick_mode: bool = False,
        timestamp_seconds: float | None = None,
        *,
        prompt_style: PromptStyle | None = None,
    ) -> ServeDetectionResult:
        """
        Detect if a single frame contains a serve.

        Args:
            frame_path: Path to the frame image.
            quick_mode: If True, use simplified prompt for faster inference.
            timestamp_seconds: Optional timestamp of the frame in the video.
            prompt_style: ``training`` = same question/YES-NO as ``serve_detection_training.ipynb``
                (default when a LoRA adapter is loaded). ``structured`` = ``SERVE_DETECTED:`` template.

        Returns:
            ServeDetectionResult with detection details.
        """
        frame_path = Path(frame_path)

        if quick_mode:
            prompt = SERVE_QUICK_PROMPT
            max_new_tokens = 256
        else:
            style = self._resolve_prompt_style(prompt_style)
            if style == "training":
                prompt = get_serve_training_style_prompt()
                max_new_tokens = 128
            else:
                prompt = get_serve_detection_prompt()
                max_new_tokens = 512

        response = self._client.analyze_image(
            image_path=frame_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )

        if quick_mode:
            is_serve = self._parse_quick_response(response)
            return ServeDetectionResult(
                frame_path=frame_path,
                is_serve=is_serve,
                confidence=Confidence.MEDIUM,
                reasoning="Quick mode detection",
                raw_response=response,
                timestamp_seconds=timestamp_seconds,
            )

        style = self._resolve_prompt_style(prompt_style)
        if style == "training":
            return self._parse_training_style_response(response, frame_path, timestamp_seconds)
        return self._parse_full_response(response, frame_path, timestamp_seconds)

    def detect_batch(
        self,
        frame_paths: list[str | Path],
        timestamps: list[float] | None = None,
        *,
        prompt_style: PromptStyle | None = None,
    ) -> list[ServeDetectionResult]:
        """
        Detect serves in multiple frames.

        Args:
            frame_paths: List of paths to frame images.
            timestamps: Optional list of timestamps for each frame.
            prompt_style: Passed through to :meth:`detect`.

        Returns:
            List of ServeDetectionResult for each frame.
        """
        results = []
        timestamps = timestamps or [None] * len(frame_paths)

        for frame_path, timestamp in zip(frame_paths, timestamps):
            result = self.detect(
                frame_path, timestamp_seconds=timestamp, prompt_style=prompt_style
            )
            results.append(result)

        return results

    def detect_sequence(
        self,
        frame_paths: list[str | Path],
    ) -> ServeSequenceResult:
        """
        Detect serve in a sequence of frames (temporal analysis).

        This method analyzes multiple frames together to leverage
        temporal context, which can improve detection accuracy.

        Args:
            frame_paths: List of paths to sequential frame images.

        Returns:
            ServeSequenceResult with detection details.
        """
        frame_paths = [Path(p) for p in frame_paths]
        prompt = get_sequence_prompt(len(frame_paths))

        # Analyze all frames together
        response = self._client.analyze_images(
            image_paths=frame_paths,
            prompt=prompt,
            max_new_tokens=512,
            temperature=0.1,
        )

        return self._parse_sequence_response(response, frame_paths)

    def detect_shortclip_window(
        self,
        frame_paths: list[str | Path],
        *,
        anchor_timestamp_seconds: float | None = None,
        prompt_style: PromptStyle | None = None,
    ) -> ServeDetectionResult:
        """
        Multi-frame verification aligned with short-clip LoRA training (Colab notebook).

        Uses one VLM call with multiple images and the short-clip YES/NO prompt when
        ``prompt_style`` is ``training`` (default when a LoRA adapter is loaded).
        """
        frame_paths = [Path(p) for p in frame_paths]
        if not frame_paths:
            return ServeDetectionResult(
                frame_path=Path("."),
                is_serve=False,
                confidence=Confidence.UNKNOWN,
                reasoning="No frames in short-clip window",
                raw_response="",
                timestamp_seconds=anchor_timestamp_seconds,
            )

        style = self._resolve_prompt_style(prompt_style)
        if style == "training":
            prompt = get_shortclip_training_style_prompt()
            max_new_tokens = 128
        else:
            prompt = get_serve_detection_prompt()
            max_new_tokens = 512

        response = self._client.analyze_images(
            frame_paths,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
        )

        if style == "training":
            return self._parse_training_style_response(
                response, frame_paths[-1], anchor_timestamp_seconds
            )
        return self._parse_full_response(response, frame_paths[-1], anchor_timestamp_seconds)

    def _parse_quick_response(self, response: str) -> bool:
        """Parse quick mode response (YES/NO only)."""
        response_upper = response.upper().strip()
        return "YES" in response_upper and "NO" not in response_upper.split("YES")[0]

    def _parse_training_style_response(
        self,
        response: str,
        frame_path: Path,
        timestamp_seconds: float | None,
    ) -> ServeDetectionResult:
        """
        Parse YES/NO replies matching ``serve_detection_training.ipynb`` / Colab test parsers.

        Expects the model to lead with YES or NO (optionally followed by explanation).
        """
        is_serve, confidence, reasoning = parse_training_style_serve_response(response)
        return ServeDetectionResult(
            frame_path=frame_path,
            is_serve=is_serve,
            confidence=confidence,
            reasoning=reasoning,
            player_position=None,
            raw_response=response,
            timestamp_seconds=timestamp_seconds,
        )

    def _parse_full_response(
        self,
        response: str,
        frame_path: Path,
        timestamp_seconds: float | None,
    ) -> ServeDetectionResult:
        """Parse full detection response."""
        # Default values
        is_serve = False
        confidence = Confidence.UNKNOWN
        reasoning = ""
        player_position = None

        # Parse SERVE_DETECTED
        serve_match = re.search(r"SERVE_DETECTED:\s*(YES|NO)", response, re.IGNORECASE)
        if serve_match:
            is_serve = serve_match.group(1).upper() == "YES"

        # Parse CONFIDENCE
        conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = Confidence(conf_match.group(1).upper())
            except ValueError:
                confidence = Confidence.UNKNOWN

        # Parse REASONING
        reason_match = re.search(r"REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        # Parse PLAYER_POSITION
        pos_match = re.search(r"PLAYER_POSITION:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL)
        if pos_match:
            player_position = pos_match.group(1).strip()

        return ServeDetectionResult(
            frame_path=frame_path,
            is_serve=is_serve,
            confidence=confidence,
            reasoning=reasoning,
            player_position=player_position,
            raw_response=response,
            timestamp_seconds=timestamp_seconds,
        )

    def _parse_sequence_response(
        self,
        response: str,
        frame_paths: list[Path],
    ) -> ServeSequenceResult:
        """Parse sequence detection response."""
        is_serve = False
        confidence = Confidence.UNKNOWN
        serve_frame_index = None
        reasoning = ""

        # Parse SERVE_DETECTED
        serve_match = re.search(r"SERVE_DETECTED:\s*(YES|NO)", response, re.IGNORECASE)
        if serve_match:
            is_serve = serve_match.group(1).upper() == "YES"

        # Parse CONFIDENCE
        conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MEDIUM|LOW)", response, re.IGNORECASE)
        if conf_match:
            try:
                confidence = Confidence(conf_match.group(1).upper())
            except ValueError:
                confidence = Confidence.UNKNOWN

        # Parse SERVE_FRAME
        frame_match = re.search(r"SERVE_FRAME:\s*(\d+|N/A)", response, re.IGNORECASE)
        if frame_match and frame_match.group(1).upper() != "N/A":
            try:
                serve_frame_index = int(frame_match.group(1)) - 1  # Convert to 0-indexed
            except ValueError:
                serve_frame_index = None

        # Parse REASONING
        reason_match = re.search(r"REASONING:\s*(.+?)(?=\n[A-Z_]+:|$)", response, re.DOTALL)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return ServeSequenceResult(
            frame_paths=frame_paths,
            is_serve=is_serve,
            confidence=confidence,
            serve_frame_index=serve_frame_index,
            reasoning=reasoning,
            raw_response=response,
        )

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload_model()
        return False


def parse_training_style_serve_response(response: str) -> tuple[bool, Confidence, str]:
    """
    Interpret model text for :data:`~pickleball_serve_detection.prompts.SERVE_TRAINING_STYLE_PROMPT`.

    Aligned with common patterns in ``serve_detection_test.ipynb`` / training-style outputs.
    """
    text = (response or "").strip()
    if not text:
        return False, Confidence.UNKNOWN, ""

    response_lower = text.lower()
    head = response_lower[:80].lstrip()

    if head.startswith("yes"):
        return True, Confidence.HIGH, text
    if head.startswith("no"):
        if "rally" not in response_lower and "returning" not in response_lower:
            if "ball in hand" in response_lower or "with the ball" in response_lower:
                return True, Confidence.MEDIUM, text
            if "preparing to serve" in response_lower or "about to serve" in response_lower:
                return True, Confidence.MEDIUM, text
        return False, Confidence.HIGH, text
    if head.startswith("maybe"):
        return True, Confidence.LOW, text

    serve_match = re.search(r"SERVE:\s*(YES|NO)", text, re.IGNORECASE)
    if serve_match:
        return serve_match.group(1).upper() == "YES", Confidence.HIGH, text

    return False, Confidence.LOW, text
