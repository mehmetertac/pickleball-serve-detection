# Pickleball serve detection

Vision-language serve detection (Qwen2.5-VL + optional LoRA), two-stage cascade (audio proposer → short-clip VLM), and frame sampling helpers.

```bash
pip install pickleball-serve-detection[detection]
```

Optional training extras: `pip install pickleball-serve-detection[training]`.

Environment variables:

- `PICKLEBALL_SERVE_ADAPTER` — directory with PEFT adapter (`adapter_config.json`), or empty to disable.
