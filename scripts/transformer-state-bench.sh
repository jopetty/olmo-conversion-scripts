#!/usr/bin/env bash
set -euo pipefail

SOURCE_BASE_DIR="/weka/oe-training-default/ai2-llm/model-ladders/state-bench"

SIZE="60M"
DATASET=""
CHINCHILLA=""
SEED=""

usage() {
    cat <<EOF
Usage: $0 --dataset DATASET [--size SIZE] [--chinchilla CHINCHILLA] [--seed SEED]

Datasets:
  integer-code--r-trivial
  integer-code--aperiodic
  integer-code--periodic

Examples:
  $0 --dataset integer-code--r-trivial
  $0 --dataset integer-code--aperiodic --chinchilla 1 --seed 0
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --size)
            SIZE="$2"
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
        --seed)
            SEED="$2"
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

case "$DATASET" in
    integer-code--r-trivial|integer-code--aperiodic|integer-code--periodic)
        ;;
    *)
        echo "Unsupported state-bench dataset: ${DATASET}" >&2
        usage >&2
        exit 1
        ;;
esac

if [[ -n "$SEED" && ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "--seed must be an integer, got: $SEED" >&2
    exit 1
fi

DATASET_INPUT_DIR="${SOURCE_BASE_DIR}/${SIZE}/transformer/${DATASET}"

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

CHINCHILLA_INPUT_DIR="${DATASET_INPUT_DIR}/${CHINCHILLA}"

if [[ ! -d "$CHINCHILLA_INPUT_DIR" ]]; then
    echo "Checkpoint base directory does not exist: ${CHINCHILLA_INPUT_DIR}" >&2
    exit 1
fi

if [[ -n "$SEED" ]]; then
    SEED_DIRS=("${CHINCHILLA_INPUT_DIR}/init_seed${SEED}")

    if [[ ! -d "${SEED_DIRS[0]}" ]]; then
        echo "Seed checkpoint directory does not exist: ${SEED_DIRS[0]}" >&2
        exit 1
    fi
else
    SEED_DIRS=()
    while IFS= read -r SEED_DIR; do
        SEED_DIR_NAME="$(basename "$SEED_DIR")"
        if [[ ! "$SEED_DIR_NAME" =~ ^init_seed[0-9]+$ ]]; then
            continue
        fi
        SEED_DIRS+=("$SEED_DIR")
    done < <(find "$CHINCHILLA_INPUT_DIR" -maxdepth 1 -type d -name 'init_seed[0-9]*')

    if [[ "${#SEED_DIRS[@]}" -eq 0 ]]; then
        echo "No init_seed checkpoint directories found in: ${CHINCHILLA_INPUT_DIR}" >&2
        exit 1
    fi

    for ((i = 0; i < ${#SEED_DIRS[@]}; i++)); do
        for ((j = i + 1; j < ${#SEED_DIRS[@]}; j++)); do
            SEED_I="${SEED_DIRS[i]##*/init_seed}"
            SEED_J="${SEED_DIRS[j]##*/init_seed}"
            if ((10#$SEED_J < 10#$SEED_I)); then
                TMP_SEED_DIR="${SEED_DIRS[i]}"
                SEED_DIRS[i]="${SEED_DIRS[j]}"
                SEED_DIRS[j]="$TMP_SEED_DIR"
            fi
        done
    done
fi

CONVERSIONS_COMPLETED=0

for INPUT_DIR in "${SEED_DIRS[@]}"; do
    OUTPUT_DIR="${INPUT_DIR}-hf"

    if [[ -z "$(find "$INPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        echo "Skipping empty checkpoint directory: ${INPUT_DIR}" >&2
        continue
    fi

    echo "Loading checkpoint from ${INPUT_DIR}"

    if ! TRUST_REMOTE_CODE=True uv run python transformer_synthetic.py \
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
