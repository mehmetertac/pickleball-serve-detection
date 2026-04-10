"""Constants and default paths for pickleball-serve-detection."""

import os
from pathlib import Path

# Repo root: src/pickleball_serve_detection/constants.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DOWNLOAD_DIR = PROJECT_ROOT / "downloaded_videos"

DEFAULT_SERVE_ADAPTER_PATH = PROJECT_ROOT / "models" / "serve_detector" / "serve-shortclip-lora" / "final_adapter"
DEFAULT_SERVE_ADAPTER_PATH_LEGACY = PROJECT_ROOT / "models" / "serve-detector-lora-v5"


def _default_serve_adapter_search_paths() -> list[Path]:
    return [
        DEFAULT_SERVE_ADAPTER_PATH,
        PROJECT_ROOT / "models" / "serve-shortclip-lora" / "final_adapter",
        PROJECT_ROOT / "models" / "serve_detector" / "serve-shortclip-lora",
        PROJECT_ROOT / "models" / "serve-shortclip-lora",
        DEFAULT_SERVE_ADAPTER_PATH_LEGACY,
        PROJECT_ROOT / "models" / "serve-detector-lora-v5",
    ]


def _is_peft_adapter_dir(p: Path) -> bool:
    return p.is_dir() and (p / "adapter_config.json").is_file()


def resolve_path_str(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolved_serve_adapter_path() -> Path | None:
    if "PICKLEBALL_SERVE_ADAPTER" in os.environ:
        raw = os.environ["PICKLEBALL_SERVE_ADAPTER"].strip()
        if not raw:
            return None
        return resolve_path_str(raw)
    for candidate in _default_serve_adapter_search_paths():
        if _is_peft_adapter_dir(candidate):
            return resolve_path_str(candidate)
    return None
