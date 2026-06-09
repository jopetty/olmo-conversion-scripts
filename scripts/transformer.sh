#!/usr/bin/env bash
set -euo pipefail

SOURCE_BASE_DIR="/weka/oe-training-default/ai2-llm/model-ladders"
TARGET_BASE_DIR="/weka/oe-training-default/ai2-llm/checkpoints/jacksonp"

MODEL="olmo3-baseline-jacksonp"
SIZE="1B"
STEP=""

STEPS_A=(0 7320 8134 15454 16268 30910 32537 61819 65073 123639 130000 130147)

usage() {
    cat <<EOF
Usage: $0 --model MODEL --size SIZE [--step STEP]

Examples:
  $0 --model olmo3-baseline-jacksonp --size 1B --step 0
  $0 --model olmo3-baseline-jacksonp --size 1B
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --size)
            SIZE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

if [[ -z "$MODEL" || -z "$SIZE" ]]; then
    echo "Missing required argument." >&2
    usage >&2
    exit 1
fi

if [[ -n "$STEP" && ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "--step must be an integer, got: $STEP" >&2
    exit 1
fi

if [[ -n "$STEP" ]]; then
    STEPS=("$STEP")
else
    STEPS=("${STEPS_A[@]}")
fi

for STEP in "${STEPS[@]}"; do
    INPUT_DIR="${SOURCE_BASE_DIR}/${MODEL}/${SIZE}/step${STEP}"
    OUTPUT_DIR="${TARGET_BASE_DIR}/${MODEL}/${SIZE}/step${STEP}"

    TRUST_REMOTE_CODE=True uv run python scripts/transformer.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --tokenizer allenai/Olmo-Hybrid-7B
done
