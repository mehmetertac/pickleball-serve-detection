"""Supported VLM model ids and coarse-window inference mode names."""

VLM_MODEL_NAMES = {
    "2b": "Qwen/Qwen2.5-VL-2B-Instruct",
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
}

SERVE_COARSE_WINDOW_INFER_MODES = ("per_frame_majority", "per_frame_any", "sequence")
