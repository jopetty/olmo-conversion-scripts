"""Compare an OLMo-core checkpoint with its Hugging Face conversion.

The script runs the same token sequences through both models and compares the
complete logits tensor at every position.  Both models are loaded on the
requested device, so the default invocation is suitable for a GPU.

Example:

    uv run python verify_conversion.py \
        --olmo-core-checkpoint /path/to/olmo-core-checkpoint \
        --hf-checkpoint /path/to/hf-checkpoint

The test sequence suite mirrors ``verify_conversion_olmc.py`` so the existing
upstream coverage is retained while this script avoids the old intermediate-
logits/vLLM workflow.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch
from transformers import AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}

FLASH_ATTENTION_BACKENDS = {"flash_2", "flash_3", "flash_4"}
MAX_P95_SAMPLES = 1_000_000


def color(text: str, code: int) -> str:
    """Color terminal output while remaining safe for redirected output."""
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR") is not None:
        return text
    return f"\033[{code}m{text}\033[0m"


# Keep this suite identical to the upstream OLMo-core verification script.
TEST_SEQUENCES = [
    ("original_10tok", [1, 2, 3, 4, 5, 100, 200, 500, 1000, 2000]),
    ("single_token", [12345]),
    ("short_5tok", [100, 200, 300, 400, 500]),
    ("medium_20tok", list(range(100, 120))),
    ("longer_50tok", list(range(1000, 1050))),
    ("longer_100tok", list(range(500, 600))),
    ("random_pattern", [42, 1337, 9999, 7777, 2468, 1357, 8642, 3141, 5926, 5358]),
    ("high_vocab", [50000, 55000, 60000, 65000, 70000, 75000, 80000, 85000, 90000, 95000]),
    ("single_1", [1]),
    ("single_100", [100]),
    ("single_1000", [1000]),
    ("single_10000", [10000]),
    ("single_50000", [50000]),
    ("pair_low", [1, 2]),
    ("pair_mid", [5000, 5001]),
    ("pair_high", [80000, 80001]),
    ("pair_mixed", [100, 90000]),
    ("triple_asc", [100, 200, 300]),
    ("triple_desc", [300, 200, 100]),
    ("triple_same", [500, 500, 500]),
    ("triple_spread", [1, 50000, 99999]),
    ("five_linear", [10, 20, 30, 40, 50]),
    ("five_exp", [1, 10, 100, 1000, 10000]),
    ("five_primes", [2, 3, 5, 7, 11]),
    ("five_fib", [1, 1, 2, 3, 5]),
    ("five_powers2", [2, 4, 8, 16, 32]),
    ("ten_linear_1", list(range(1, 11))),
    ("ten_linear_100", list(range(100, 110))),
    ("ten_linear_1000", list(range(1000, 1010))),
    ("ten_linear_10000", list(range(10000, 10010))),
    ("ten_evens", list(range(2, 22, 2))),
    ("ten_odds", list(range(1, 21, 2))),
    ("ten_squares", [i**2 for i in range(1, 11)]),
    ("ten_cubes", [i**3 for i in range(1, 11)]),
    ("ten_random_a", [3847, 9182, 4756, 2938, 8471, 1029, 5738, 4829, 7364, 2918]),
    ("ten_random_b", [12847, 38291, 47562, 82934, 19283, 57382, 29384, 83921, 47283, 92831]),
    ("ten_random_c", [61234, 72345, 83456, 94567, 15678, 26789, 37890, 48901, 59012, 60123]),
    ("ten_alternating", [100, 90000, 200, 80000, 300, 70000, 400, 60000, 500, 50000]),
    ("fifteen_linear", list(range(500, 515))),
    ("fifteen_spread", list(range(0, 75000, 5000))),
    ("fifteen_random", [2847, 19283, 38472, 57261, 8374, 94827, 12938, 47382, 83927, 29384, 58273, 17384, 92837, 48273, 73829]),
    ("twenty_low", list(range(1, 21))),
    ("twenty_mid", list(range(40000, 40020))),
    ("twenty_high", list(range(90000, 90020))),
    ("twenty_spread", list(range(0, 100000, 5000))),
    ("twenty_random_a", [i * 4937 % 100000 for i in range(20)]),
    ("twenty_random_b", [i * 7919 % 100000 for i in range(20)]),
    ("twentyfive_linear", list(range(2000, 2025))),
    ("twentyfive_random", [i * 6151 % 100000 for i in range(25)]),
    ("thirty_linear", list(range(3000, 3030))),
    ("thirty_spread", list(range(0, 90000, 3000))),
    ("thirty_random", [i * 8123 % 100000 for i in range(30)]),
    ("forty_linear", list(range(4000, 4040))),
    ("forty_random", [i * 9311 % 100000 for i in range(40)]),
    ("fifty_linear_a", list(range(5000, 5050))),
    ("fifty_linear_b", list(range(50000, 50050))),
    ("fifty_random_a", [i * 3571 % 100000 for i in range(50)]),
    ("fifty_random_b", [i * 7333 % 100000 for i in range(50)]),
    ("sixtyfour_linear", list(range(6400, 6464))),
    ("sixtyfour_random", [i * 4519 % 100000 for i in range(64)]),
    ("seventyfive_linear", list(range(7500, 7575))),
    ("seventyfive_random", [i * 5347 % 100000 for i in range(75)]),
    ("hundred_linear_a", list(range(100, 200))),
    ("hundred_linear_b", list(range(10000, 10100))),
    ("hundred_linear_c", list(range(80000, 80100))),
    ("hundred_random_a", [i * 2671 % 100000 for i in range(100)]),
    ("hundred_random_b", [i * 8887 % 100000 for i in range(100)]),
    ("onetwentyeight_linear", list(range(12800, 12928))),
    ("onetwentyeight_random", [i * 6947 % 100000 for i in range(128)]),
    ("onefifty_linear", list(range(15000, 15150))),
    ("onefifty_random", [i * 4201 % 100000 for i in range(150)]),
    ("twohundred_linear", list(range(20000, 20200))),
    ("twohundred_random", [i * 7687 % 100000 for i in range(200)]),
    ("twofiftysix_linear", list(range(25600, 25856))),
    ("twofiftysix_random", [i * 3389 % 100000 for i in range(256)]),
    ("repeat_10x10", [100] * 10),
    ("repeat_50x5", [5000] * 50),
    ("repeat_100x3", [30000] * 100),
    ("zigzag_20", [100 if i % 2 == 0 else 90000 for i in range(20)]),
    ("zigzag_50", [1000 if i % 2 == 0 else 80000 for i in range(50)]),
    ("sawtooth_30", [(i % 10) * 1000 for i in range(30)]),
    ("sawtooth_60", [(i % 20) * 500 for i in range(60)]),
    ("near_zero", list(range(0, 10))),
    ("low_range", list(range(50, 100))),
    ("mid_range", list(range(49950, 50050))),
    ("high_range", list(range(99900, 100000))),
    ("mixed_ranges_20", [i * 5000 for i in range(20)]),
    ("mixed_ranges_50", [i * 2000 for i in range(50)]),
    (
        "scattered_25",
        [
            1, 1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000,
            80000, 90000, 95000, 99000, 99500, 99900, 99950, 99990, 99995,
            99999, 500, 5500, 15000, 25000, 35000,
        ],
    ),
    ("stress_512", [i * 193 % 100000 for i in range(512)]),
    ("stress_768", [i * 277 % 100000 for i in range(768)]),
    ("stress_1024", [i * 331 % 100000 for i in range(1024)]),
    ("repeat_256_pattern", [1000, 2000, 3000, 4000] * 64),
]


def load_olmo_core_model(
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype,
    attention_backend: str,
) -> tuple[Any, int]:
    """Build an OLMo-core model and load its distributed checkpoint weights."""
    from olmo_core.distributed.checkpoint import load_model_and_optim_state
    from olmo_core.nn.transformer.config import TransformerConfig

    config_path = f"{checkpoint_dir}/config.json"
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"OLMo-core config not found at {config_path}")

    with Path(config_path).open("r", encoding="utf-8") as config_file:
        experiment_config = json.load(config_file)

    # Do not mutate the checkpoint config: these fields describe distributed
    # training/runtime settings rather than the Transformer architecture.
    transformer_config_dict = dict(experiment_config["model"])
    for key in ("compile", "dp_config", "tp_config", "float8_config"):
        transformer_config_dict.pop(key, None)

    if attention_backend != "config":
        replaced_backends = replace_attention_backends(
            transformer_config_dict, attention_backend
        )
        log.info(
            "Using attention backend '%s' for %d configured attention module(s)",
            attention_backend,
            replaced_backends,
        )

    model_config = TransformerConfig.from_dict(transformer_config_dict)
    log.info("Building OLMo-core model (vocab size %d)...", model_config.vocab_size)
    model = model_config.build(init_device="meta")
    model.to_empty(device=device)

    with TemporaryDirectory() as work_dir:
        model_and_optim_dir = str(Path(checkpoint_dir) / "model_and_optim")
        load_model_and_optim_state(model_and_optim_dir, model, work_dir=work_dir)

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, model_config.vocab_size


def replace_attention_backends(config: dict[str, Any], backend: str) -> int:
    """Replace FlashAttention backend names in a serialized model config."""
    replacements = 0
    for key, value in config.items():
        if key == "backend" and value in FLASH_ATTENTION_BACKENDS:
            config[key] = backend
            replacements += 1
        elif isinstance(value, dict):
            replacements += replace_attention_backends(value, backend)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    replacements += replace_attention_backends(item, backend)
    return replacements


def load_hf_model(
    checkpoint_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, int]:
    """Load a Hugging Face causal language model on the requested device."""
    log.info("Loading Hugging Face model from %s...", checkpoint_dir)
    model: Any = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        dtype=dtype,
    )
    model.to(device=device)
    model.eval()
    vocab_size = model.config.vocab_size
    log.info("Hugging Face model loaded (vocab size %d).", vocab_size)
    return model, vocab_size


def get_logits(output: object, model_name: str) -> torch.Tensor:
    """Normalize tensor and ModelOutput-style forward results to logits."""
    if isinstance(output, torch.Tensor):
        return output

    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits

    raise TypeError(f"{model_name} forward pass did not return a logits tensor")


def compare_logits(
    core_logits: torch.Tensor,
    hf_logits: torch.Tensor,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Return numerical and top-1 agreement metrics for one sequence."""
    core_logits = core_logits.detach().float().cpu()
    hf_logits = hf_logits.detach().float().cpu()

    if core_logits.shape != hf_logits.shape:
        return {
            "shape_match": False,
            "allclose": False,
            "core_shape": list(core_logits.shape),
            "hf_shape": list(hf_logits.shape),
        }

    difference = (core_logits - hf_logits).abs()
    p95_values = difference.flatten()
    if p95_values.numel() > MAX_P95_SAMPLES:
        stride = math.ceil(p95_values.numel() / MAX_P95_SAMPLES)
        p95_values = p95_values[::stride]
    core_top1 = core_logits.argmax(dim=-1)
    hf_top1 = hf_logits.argmax(dim=-1)
    top1_matches = int((core_top1 == hf_top1).sum().item())
    positions = core_top1.numel()

    return {
        "shape_match": True,
        "allclose": bool(torch.allclose(core_logits, hf_logits, atol=atol, rtol=rtol)),
        "max_abs_diff": float(difference.max().item()),
        "mean_abs_diff": float(difference.mean().item()),
        "p95_abs_diff": float(torch.quantile(p95_values, 0.95).item()),
        "p95_sample_count": p95_values.numel(),
        "top1_matches": top1_matches,
        "top1_positions": positions,
        "top1_rate": 100.0 * top1_matches / positions if positions else 100.0,
    }


def run(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")

    dtype = DTYPES[args.dtype]
    sequences = TEST_SEQUENCES[: args.max_sequences] if args.max_sequences else TEST_SEQUENCES

    log.info("Using device=%s dtype=%s", device, args.dtype)
    log.info("Testing %d sequences", len(sequences))
    log.info("Loading OLMo-core checkpoint...")
    core_model, core_vocab_size = load_olmo_core_model(
        args.olmo_core_checkpoint, device, dtype, args.attention_backend
    )
    log.info("Loading Hugging Face checkpoint...")
    hf_model, hf_vocab_size = load_hf_model(args.hf_checkpoint, device, dtype)

    if core_vocab_size != hf_vocab_size:
        log.warning(
            "Vocabulary sizes differ: OLMo-core=%d, Hugging Face=%d. "
            "Only the shared vocabulary can be compared.",
            core_vocab_size,
            hf_vocab_size,
        )
    vocab_size = min(core_vocab_size, hf_vocab_size)

    results: list[dict[str, Any]] = []
    skipped = 0
    try:
        for index, (name, sequence) in enumerate(sequences, start=1):
            input_ids_list = [token for token in sequence if 0 <= token < vocab_size]
            if not input_ids_list:
                skipped += 1
                log.warning("[%d/%d] %s: skipped (no valid input tokens)", index, len(sequences), name)
                continue
            if len(input_ids_list) != len(sequence):
                log.warning(
                    "[%d/%d] %s: filtered %d token(s) outside the shared vocabulary",
                    index,
                    len(sequences),
                    name,
                    len(sequence) - len(input_ids_list),
                )

            input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
            log.info("[%d/%d] %s (length=%d)", index, len(sequences), name, len(input_ids_list))

            with torch.inference_mode():
                core_logits = get_logits(core_model(input_ids), "OLMo-core")
                hf_logits = get_logits(hf_model(input_ids), "Hugging Face")

            result = {"name": name, "sequence_length": len(input_ids_list)}
            result.update(
                compare_logits(
                    core_logits[0],
                    hf_logits[0],
                    atol=args.atol,
                    rtol=args.rtol,
                )
            )
            results.append(result)

            if result["shape_match"]:
                log.info(
                    "    %s | top-1 %.2f%% | mean abs %.6g | max abs %.6g",
                    "MATCH" if result["allclose"] else "MISMATCH",
                    result["top1_rate"],
                    result["mean_abs_diff"],
                    result["max_abs_diff"],
                )
            else:
                log.error(
                    "    MISMATCH | shape core=%s hf=%s",
                    result["core_shape"],
                    result["hf_shape"],
                )

            del core_logits, hf_logits, input_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        del core_model, hf_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No sequences were available for comparison")

    total_positions = sum(int(result.get("top1_positions", 0)) for result in results)
    total_top1_matches = sum(int(result.get("top1_matches", 0)) for result in results)
    allclose = all(bool(result["allclose"]) for result in results)
    shape_match = all(bool(result["shape_match"]) for result in results)
    sequence_passes = sum(
        bool(
            result["shape_match"]
            and result["allclose"]
            and result.get("top1_matches", 0) == result.get("top1_positions", 0)
        )
        for result in results
    )
    sequence_failures = len(results) - sequence_passes
    top1_rate = 100.0 * total_top1_matches / total_positions if total_positions else 0.0
    max_abs_diff = max(float(result.get("max_abs_diff", float("inf"))) for result in results)
    mean_abs_diff = sum(float(result.get("mean_abs_diff", float("inf"))) for result in results) / len(results)

    passed = shape_match and allclose and total_top1_matches == total_positions
    pass_squares = color("■" * sequence_passes, 32)
    fail_squares = color("■" * sequence_failures, 31)
    sequence_bar = pass_squares + fail_squares
    report = {
        "passed": passed,
        "olmo_core_checkpoint": args.olmo_core_checkpoint,
        "hf_checkpoint": args.hf_checkpoint,
        "device": str(device),
        "dtype": args.dtype,
        "atol": args.atol,
        "rtol": args.rtol,
        "core_vocab_size": core_vocab_size,
        "hf_vocab_size": hf_vocab_size,
        "sequences_tested": len(results),
        "sequences_skipped": skipped,
        "sequence_passes": sequence_passes,
        "sequence_failures": sequence_failures,
        "positions_tested": total_positions,
        "top1_matches": total_top1_matches,
        "top1_rate": top1_rate,
        "max_abs_diff": max_abs_diff,
        "mean_sequence_abs_diff": mean_abs_diff,
        "sequence_results": results,
    }

    print("\n" + "=" * 72)
    print("CONVERSION VERIFICATION")
    print("=" * 72)
    print(f"Sequences:       {len(results)} tested, {skipped} skipped")
    print(f"Test results:    {sequence_bar}")
    print(
        f"                 {color(f'{sequence_passes} passed', 32)} / "
        f"{color(f'{sequence_failures} failed', 31)}"
    )
    print(f"Positions:       {total_top1_matches}/{total_positions} top-1 matches ({top1_rate:.2f}%)")
    print(f"Max abs diff:     {max_abs_diff:.6g}")
    print(f"Mean abs diff:    {mean_abs_diff:.6g}")
    print(f"Allclose:         {allclose} (atol={args.atol}, rtol={args.rtol})")
    result_text = color("PASS", 32) if passed else color("FAIL", 31)
    print(f"RESULT:           {result_text}")
    print("=" * 72)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        log.info("JSON report written to %s", output_path)

    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--olmo-core-checkpoint",
        "--checkpoint",
        dest="olmo_core_checkpoint",
        required=True,
        help="Path to the OLMo-core checkpoint directory",
    )
    parser.add_argument(
        "--hf-checkpoint",
        "--model-path",
        dest="hf_checkpoint",
        required=True,
        help="Path to the Hugging Face-format checkpoint directory",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for both models (default: cuda)",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
        help="Inference dtype for both models (default: bfloat16)",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["config", "torch", "flash_2", "flash_3", "flash_4"],
        default="torch",
        help=(
            "OLMo-core attention backend. 'torch' works on all GPUs and is the "
            "default; 'config' preserves the checkpoint setting."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for the full-logit allclose check (default: 0.01)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for the full-logit allclose check (default: 0.01)",
    )
    parser.add_argument(
        "--max-sequences",
        type=int,
        default=None,
        help="Only run the first N sequences (useful for a quick smoke test)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path for a JSON report",
    )
    args = parser.parse_args()
    if args.max_sequences is not None and args.max_sequences < 1:
        parser.error("--max-sequences must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("--atol and --rtol must be non-negative")

    from olmo_core.utils import prepare_cli_environment

    prepare_cli_environment()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
