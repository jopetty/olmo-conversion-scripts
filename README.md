## Verify a checkpoint conversion

Run the OLMo-core and Hugging Face checkpoints on the same token sequences and
compare their full forward-pass logits:

```bash
uv run python verify_conversion.py \
  --olmo-core-checkpoint /path/to/olmo-core-checkpoint \
  --hf-checkpoint /path/to/hf-checkpoint
```

The command defaults to CUDA and bfloat16, reports full-logit allclose and
per-position top-1 agreement, and exits with status 1 if the conversion does
not match. Use `--max-sequences 1` for a quick smoke test or `--output
verification.json` to save the detailed results.
