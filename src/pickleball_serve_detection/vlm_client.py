"""Vision-Language Model client for Qwen2.5-VL inference."""

from pathlib import Path
from typing import Any

# Lazy imports to avoid loading heavy dependencies until needed
_transformers_available = None
_torch_available = None


def _check_dependencies() -> dict[str, bool]:
    """Check which dependencies are available."""
    global _transformers_available, _torch_available

    if _transformers_available is None:
        try:
            import transformers  # noqa: F401

            _transformers_available = True
        except ImportError:
            _transformers_available = False

    if _torch_available is None:
        try:
            import torch  # noqa: F401

            _torch_available = True
        except ImportError:
            _torch_available = False

    return {
        "transformers": _transformers_available,
        "torch": _torch_available,
    }


class VLMClient:
    """Client for Vision-Language Model inference using Qwen2.5-VL."""

    # Supported model variants
    SUPPORTED_MODELS = [
        "Qwen/Qwen2.5-VL-2B-Instruct",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
    ]

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "auto",
        load_in_4bit: bool = True,
        torch_dtype: str = "auto",
        adapter_path: str | Path | None = None,
    ):
        """
        Initialize the VLM client.

        Args:
            model_name: HuggingFace model name or local path.
            device: Device to run inference on ('auto', 'cuda', 'cpu').
            load_in_4bit: Whether to use 4-bit quantization (reduces memory).
            torch_dtype: Data type for model weights ('auto', 'float16', 'bfloat16').
            adapter_path: Optional PEFT LoRA adapter directory (adapter_config.json).
        """
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.torch_dtype = torch_dtype
        self.adapter_path = Path(adapter_path).resolve() if adapter_path else None

        self._model = None
        self._processor = None
        self._is_loaded = False

    def _ensure_dependencies(self) -> None:
        """Ensure required dependencies are available."""
        deps = _check_dependencies()
        missing = [name for name, available in deps.items() if not available]

        if missing:
            raise ImportError(
                f"Missing required dependencies: {', '.join(missing)}. "
                f"Install them with: pip install transformers torch"
            )

    def load_model(self) -> None:
        """Load the model and processor into memory."""
        if self._is_loaded:
            return

        self._ensure_dependencies()

        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        # Determine torch dtype
        if self.torch_dtype == "auto":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif self.torch_dtype == "float16":
            dtype = torch.float16
        elif self.torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32

        # Load configuration based on quantization setting
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self.device,
        }

        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
            except ImportError:
                print(
                    "Warning: bitsandbytes not available, loading without quantization. "
                    "Install with: pip install bitsandbytes"
                )

        print(f"Loading model: {self.model_name}...")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, **load_kwargs
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

        if self.adapter_path is not None:
            if not self.adapter_path.is_dir():
                raise FileNotFoundError(f"LoRA adapter directory not found: {self.adapter_path}")
            try:
                from peft import PeftModel
            except ImportError as e:
                raise ImportError(
                    "PEFT is required to load LoRA adapters. "
                    "Install with: pip install pickleball-serve-detection[training]"
                ) from e
            print(f"Loading LoRA adapter: {self.adapter_path}...")
            self._model = PeftModel.from_pretrained(self._model, str(self.adapter_path))
            self._model.eval()

        self._is_loaded = True
        print("Model loaded successfully!")

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        self._is_loaded = False

        # Clear CUDA cache if available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def analyze_image(
        self,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        Analyze a single image with the given prompt.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt for the analysis.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            Model's text response.
        """
        if not self._is_loaded:
            self.load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Build message format for Qwen2.5-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path.absolute())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        return self._generate(messages, max_new_tokens, temperature)

    def analyze_images(
        self,
        image_paths: list[str | Path],
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
    ) -> str:
        """
        Analyze multiple images with the given prompt.

        Args:
            image_paths: List of paths to image files.
            prompt: Text prompt for the analysis.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Model's text response.
        """
        if not self._is_loaded:
            self.load_model()

        # Build content with multiple images
        content: list[dict[str, str]] = []
        for img_path in image_paths:
            img_path = Path(img_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {img_path}")
            content.append({"type": "image", "image": str(img_path.absolute())})

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        return self._generate(messages, max_new_tokens, temperature)

    def _generate(
        self,
        messages: list[dict],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        """Internal method to generate response from messages."""
        from qwen_vl_utils import process_vision_info

        # Apply chat template
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process images/videos
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self._model.device)

        # Generate response
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
        )

        # Decode response (remove input tokens)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return response

    def __enter__(self):
        """Context manager entry - load model."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload model."""
        self.unload_model()
        return False
