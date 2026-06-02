#!/bin/bash

TRUST_REMOTE_CODE=True uv run python yashas_hybrid.py \
    --input_dir /weka/oe-training-default/ai2-llm/model-ladders/olmo3-hybrid-gdn-deux/1B/step0 \
    --output_dir /weka/oe-training-default/ai2-llm/checkpoints/jacksonp/olmo3-hybrid-gdn-deux/1B/step0 \
    --tokenizer allenai/Olmo-Hybrid-7B
