#!/usr/bin/env bash
set -euo pipefail

SOURCE_BASE_DIR="/weka/oe-training-default/ai2-llm/checkpoints/yashasbls/hybrid-small-transformer-275M"
TARGET_BASE_DIR="/weka/oe-training-default/ai2-llm/checkpoints/jacksonp"

MODEL="transformer-small-Cx100"
SIZE="275M"
STEP=""

usage() {
    cat <<EOF
Usage: $0 --model MODEL --size SIZE [--step STEP]

Examples:
  $0 --step 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
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

if [[ -z "$SIZE" ]]; then
    echo "Missing required argument." >&2
    usage >&2
    exit 1
fi

if [[ -n "$STEP" && ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "--step must be an integer, got: $STEP" >&2
    exit 1
fi

INPUT_DIR="${SOURCE_BASE_DIR}/step${STEP}"
OUTPUT_DIR="${TARGET_BASE_DIR}/${MODEL}/${SIZE}/step${STEP}"

TRUST_REMOTE_CODE=True uv run python scripts/transformer-yashas.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR"
