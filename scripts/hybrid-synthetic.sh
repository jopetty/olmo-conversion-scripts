#!/usr/bin/env bash
set -euo pipefail

SOURCE_BASE_DIR="/weka/oe-training-default/ai2-llm/model-ladders/synthetic-ladder"
TARGET_BASE_DIR="/weka/oe-training-default/ai2-llm/checkpoints/jacksonp"

SIZE="60M"
DATASET=""
CHINCHILLA=""
STEP=""

usage() {
    cat <<EOF
Usage: $0 --dataset DATASET [--size SIZE] [--step STEP] [--chinchilla CHINCHILLA]

Examples:
  $0 --dataset aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2_lt512
  $0 --dataset aperiodic_unsupervised_n200000000_v26_a50_m64_z1p2_s2_lt512 --step 0
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
        --dataset)
            DATASET="$2"
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

if [[ -z "$SIZE" || -z "$DATASET" ]]; then
    echo "Missing required argument." >&2
    usage >&2
    exit 1
fi

if [[ -n "$STEP" && ! "$STEP" =~ ^[0-9]+$ ]]; then
    echo "--step must be an integer, got: $STEP" >&2
    exit 1
fi

DATASET_INPUT_DIR="${SOURCE_BASE_DIR}/${SIZE}/hybrid/${DATASET}"

if [[ ! -d "$DATASET_INPUT_DIR" ]]; then
    echo "Dataset checkpoint directory does not exist: ${DATASET_INPUT_DIR}" >&2
    exit 1
fi

if [[ -n "$CHINCHILLA" ]]; then
    if [[ "$CHINCHILLA" != Cx* ]]; then
        CHINCHILLA="Cx${CHINCHILLA}"
    fi
else
    CHINCHILLA_DIRS=()
    while IFS= read -r CHINCHILLA_DIR; do
        CHINCHILLA_DIRS+=("$(basename "$CHINCHILLA_DIR")")
    done < <(find "$DATASET_INPUT_DIR" -maxdepth 1 -type d -name 'Cx*' | sort)

    if [[ "${#CHINCHILLA_DIRS[@]}" -eq 0 ]]; then
        echo "No Cx checkpoint directories found in: ${DATASET_INPUT_DIR}" >&2
        exit 1
    fi

    if [[ "${#CHINCHILLA_DIRS[@]}" -gt 1 ]]; then
        echo "Multiple Cx checkpoint directories found in ${DATASET_INPUT_DIR}: ${CHINCHILLA_DIRS[*]}" >&2
        echo "Pass --chinchilla explicitly." >&2
        exit 1
    fi

    CHINCHILLA="${CHINCHILLA_DIRS[0]}"
    echo "Inferred --chinchilla ${CHINCHILLA} from ${DATASET_INPUT_DIR}"
fi

BASE_INPUT_DIR="${DATASET_INPUT_DIR}/${CHINCHILLA}"

if [[ ! -d "$BASE_INPUT_DIR" ]]; then
    echo "Checkpoint base directory does not exist: ${BASE_INPUT_DIR}" >&2
    exit 1
fi

if [[ -n "$STEP" ]]; then
    STEP_DIRS=("${BASE_INPUT_DIR}/step${STEP}")
else
    STEP_DIRS=()
    while IFS= read -r STEP_DIR; do
        STEP_DIR_NAME="$(basename "$STEP_DIR")"
        if [[ ! "$STEP_DIR_NAME" =~ ^step[0-9]+$ ]]; then
            continue
        fi
        STEP_DIRS+=("$STEP_DIR")
    done < <(find "$BASE_INPUT_DIR" -maxdepth 1 -type d -name 'step[0-9]*')

    if [[ "${#STEP_DIRS[@]}" -eq 0 ]]; then
        echo "No checkpoint step directories found in: ${BASE_INPUT_DIR}" >&2
        exit 1
    fi

    for ((i = 0; i < ${#STEP_DIRS[@]}; i++)); do
        for ((j = i + 1; j < ${#STEP_DIRS[@]}; j++)); do
            STEP_I="${STEP_DIRS[i]##*/step}"
            STEP_J="${STEP_DIRS[j]##*/step}"
            if ((10#$STEP_J < 10#$STEP_I)); then
                TMP_STEP_DIR="${STEP_DIRS[i]}"
                STEP_DIRS[i]="${STEP_DIRS[j]}"
                STEP_DIRS[j]="$TMP_STEP_DIR"
            fi
        done
    done
fi

CONVERSIONS_COMPLETED=0

for INPUT_DIR in "${STEP_DIRS[@]}"; do
    STEP_DIR_NAME="$(basename "$INPUT_DIR")"
    STEP="${STEP_DIR_NAME#step}"
    OUTPUT_DIR="${TARGET_BASE_DIR}/hybrid-${DATASET}-${CHINCHILLA}/${SIZE}/step${STEP}"

    if [[ ! -d "$INPUT_DIR" ]]; then
        echo "Skipping missing checkpoint directory: ${INPUT_DIR}" >&2
        continue
    fi

    if [[ -z "$(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Skipping empty checkpoint directory: ${INPUT_DIR}" >&2
        continue
    fi

    echo "Loading checkpoint from ${INPUT_DIR}"

    if ! TRUST_REMOTE_CODE=True uv run python hybrid_synthetic.py \
        --input_dir "$INPUT_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --tokenizer allenai/Olmo-Hybrid-7B; then
        echo "Conversion failed for ${INPUT_DIR}; continuing." >&2
        continue
    fi

    CONVERSIONS_COMPLETED=$((CONVERSIONS_COMPLETED + 1))
done

if [[ "$CONVERSIONS_COMPLETED" -eq 0 ]]; then
    echo "No checkpoints were converted successfully." >&2
    exit 1
fi
