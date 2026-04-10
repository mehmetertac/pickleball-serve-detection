"""Pickleball serve detection: VLM, cascade, frame sampling."""

from pickleball_serve_detection.audio_proposer import (
    propose_serves_by_gap,
    propose_serves_by_onset,
)
from pickleball_serve_detection.cascade import (
    CascadeConfig,
    extract_frames_for_window,
    merge_nearby_timestamps,
    parse_external_candidates,
    run_cascade,
)
from pickleball_serve_detection.coarse_window import aggregate_coarse_window_serve_votes
from pickleball_serve_detection.constants import resolved_serve_adapter_path
from pickleball_serve_detection.frame_sampler import (
    COARSE_MANIFEST_VERSION,
    FrameSampler,
    SampledFrame,
    load_coarse_serve_manifest,
    save_coarse_serve_manifest,
    sample_window_with_offset,
    serve_training_style_window_times,
)
from pickleball_serve_detection.model_names import (
    SERVE_COARSE_WINDOW_INFER_MODES,
    VLM_MODEL_NAMES,
)
from pickleball_serve_detection.prompts import SERVE_TRAINING_STYLE_PROMPT
from pickleball_serve_detection.serve_detector import (
    ServeDetector,
    parse_training_style_serve_response,
)
from pickleball_serve_detection.serve_events import serve_timestamps_from_detections
from pickleball_serve_detection.vlm_client import VLMClient

__version__ = "0.1.0"

__all__ = [
    "COARSE_MANIFEST_VERSION",
    "CascadeConfig",
    "FrameSampler",
    "SERVE_COARSE_WINDOW_INFER_MODES",
    "SERVE_TRAINING_STYLE_PROMPT",
    "SampledFrame",
    "ServeDetector",
    "VLMClient",
    "VLM_MODEL_NAMES",
    "aggregate_coarse_window_serve_votes",
    "extract_frames_for_window",
    "load_coarse_serve_manifest",
    "merge_nearby_timestamps",
    "parse_external_candidates",
    "parse_training_style_serve_response",
    "propose_serves_by_gap",
    "propose_serves_by_onset",
    "resolved_serve_adapter_path",
    "run_cascade",
    "sample_window_with_offset",
    "save_coarse_serve_manifest",
    "serve_timestamps_from_detections",
    "serve_training_style_window_times",
]
