#!/usr/bin/env bash
set -euo pipefail

SOURCE_BASE_DIR="/weka/oe-training-default/ai2-llm/model-ladders/sensitivity-ladder"
TARGET_BASE_DIR="/weka/oe-training-default/ai2-llm/checkpoints/jacksonp"

SIZE=""
STEP=""
MIXIN=""
CHINCHILLA=""

usage() {
    cat <<EOF
Usage: $0 --size SIZE --step STEP --mixin MIXIN --chinchilla CHINCHILLA

Examples:
  $0 --size 275M --step 0 --mixin aperiodic_supervised_n10000_v26_a50_m64_z1p2_s3  --chinchilla 8
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)
            SIZE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --mixin)
            MIXIN="$2"
            shift 2
            ;;
        --chinchilla)
            CHINCHILLA="$2"
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

if [[ -z "$SIZE" || -z "$STEP" || -z "$MIXIN" || -z "$CHINCHILLA" ]]; then
    echo "Missing required argument." >&2
    usage >&2
    exit 1
fi

if [[ ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "--step must be an integer, got: $STEP" >&2
    exit 1
fi

INPUT_DIR="${SOURCE_BASE_DIR}/${SIZE}/transformer/${MIXIN}/Cx${CHINCHILLA}/step${STEP}"
OUTPUT_DIR="${TARGET_BASE_DIR}/transformer-${MIXIN}-Cx${CHINCHILLA}/${SIZE}/step${STEP}"

echo "Loading checkpoint from ${INPUT_DIR}"

INPUT_DIR="${SOURCE_BASE_DIR}/${MODEL}/${SIZE}/step${STEP}"
OUTPUT_DIR="${TARGET_BASE_DIR}/${MODEL}/${SIZE}/step${STEP}"

TRUST_REMOTE_CODE=True uv run python scripts/transformer.py \
    --input_dir "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --tokenizer allenai/Olmo-Hybrid-7B
