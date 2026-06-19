from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer, OlmoHybridSmallConfig

from hybrid import DTYPE_MAP, load_model


def _get_tokenizer_config(config: dict[str, Any]) -> dict[str, Any]:
    if "dataset" in config:
        return config["dataset"]["tokenizer"]

    instance_sources = config.get("instance_sources", [])
    for instance_source in instance_sources:
        for source in instance_source.get("sources", []):
            if "tokenizer" in source:
                return source["tokenizer"]

    raise KeyError("Could not find tokenizer config under 'dataset.tokenizer' or 'instance_sources'.")


def _get_max_sequence_length(config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override

    max_sequence_length = config.get("train_module", {}).get("max_sequence_length")
    if max_sequence_length is not None:
        return max_sequence_length

    max_sequence_length = config.get("dataset", {}).get("sequence_length")
    if max_sequence_length is not None:
        return max_sequence_length

    instance_sources = config.get("instance_sources", [])
    sequence_lengths = [source["sequence_length"] for source in instance_sources if "sequence_length" in source]
    if sequence_lengths:
        return max(sequence_lengths)

    max_sequence_length = 8192
    print(f"Warning: max_sequence_length not found in config or CLI, using default: {max_sequence_length}")
    return max_sequence_length


def _required_any(loaded: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    for key in keys:
        if key in loaded:
            return loaded[key]
    raise KeyError(f"Missing expected checkpoint key. Tried: {keys}")


def _layer_weight(loaded: dict[str, torch.Tensor], layer_i: int, suffix: str) -> torch.Tensor:
    return _required_any(
        loaded,
        [
            f"blocks.{layer_i}.attention.{suffix}",
            f"blocks.{layer_i}.sequence_mixer.{suffix}",
        ],
    )


def _convert_attention_layer_weights(
    loaded: dict[str, torch.Tensor],
    layer_i: int,
) -> dict[str, torch.Tensor]:
    hf_prefix = f"model.layers.{layer_i}"
    return {
        f"{hf_prefix}.self_attn.q_proj.weight": _layer_weight(loaded, layer_i, "w_q.weight"),
        f"{hf_prefix}.self_attn.k_proj.weight": _layer_weight(loaded, layer_i, "w_k.weight"),
        f"{hf_prefix}.self_attn.v_proj.weight": _layer_weight(loaded, layer_i, "w_v.weight"),
        f"{hf_prefix}.self_attn.o_proj.weight": _layer_weight(loaded, layer_i, "w_out.weight"),
        f"{hf_prefix}.self_attn.attn_gate.weight": _layer_weight(loaded, layer_i, "w_g.weight"),
        f"{hf_prefix}.self_attn.q_norm.weight": _layer_weight(loaded, layer_i, "q_norm.weight"),
        f"{hf_prefix}.self_attn.k_norm.weight": _layer_weight(loaded, layer_i, "k_norm.weight"),
        f"{hf_prefix}.mlp.gate_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w1.weight"],
        f"{hf_prefix}.mlp.down_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w2.weight"],
        f"{hf_prefix}.mlp.up_proj.weight": loaded[f"blocks.{layer_i}.feed_forward.w3.weight"],
        f"{hf_prefix}.input_layernorm.weight": loaded[f"blocks.{layer_i}.attention_norm.weight"],
        f"{hf_prefix}.post_attention_layernorm.weight": loaded[f"blocks.{layer_i}.post_attention_norm.weight"],
        f"{hf_prefix}.ffn_layernorm.weight": loaded[f"blocks.{layer_i}.feed_forward_norm.weight"],
        f"{hf_prefix}.post_feedforward_layernorm.weight": loaded[
            f"blocks.{layer_i}.post_feed_forward_norm.weight"
        ],
    }


def _write_tokenizer(
    output_path: str,
    tokenizer_id: str | None,
    tokenizer_config: dict[str, Any],
    max_sequence_length: int,
) -> None:
    tokenizer_id = tokenizer_id or tokenizer_config.get("identifier")
    if not tokenizer_id:
        print("Warning: No tokenizer identifier found in config. Skipping tokenizer save.")
        return

    print(f"Saving tokenizer '{tokenizer_id}' to {output_path}.")
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(tokenizer_id))
    tokenizer.model_max_length = max_sequence_length
    tokenizer.pad_token_id = tokenizer_config.get("pad_token_id")
    tokenizer.bos_token_id = tokenizer_config.get("bos_token_id")
    tokenizer.eos_token_id = tokenizer_config.get("eos_token_id")
    tokenizer.save_pretrained(output_path)


def _patch_nope_config(model_path: str) -> None:
    config_path = Path(model_path) / "config.json"
    config_dict = json.loads(config_path.read_text())
    config_dict["rope_parameters"] = {"rope_theta": None}
    config_dict.pop("rope_scaling", None)
    config_dict.pop("rope_theta", None)
    config_path.write_text(json.dumps(config_dict, indent=2))
    print("Patched config.json: rope_parameters={'rope_theta': null} (NoPE)")


def write_model(
    model_path: str,
    input_base_path: str,
    include_tokenizer: bool = True,
    tokenizer_id: str | None = None,
    max_sequence_length: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    os.makedirs(model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.json"
    olmo_config = json.loads(config_path.read_text())
    model_config = olmo_config["model"]
    block_config = model_config["block"]
    attention_config = block_config["sequence_mixer"]
    feed_forward_config = block_config["feed_forward"]
    tokenizer_config = _get_tokenizer_config(olmo_config)

    n_layers = model_config["n_layers"]
    dim = model_config["d_model"]
    n_heads = attention_config["n_heads"]
    n_kv_heads = attention_config.get("n_kv_heads", n_heads)
    head_dim = attention_config.get("head_dim", dim // n_heads)
    max_sequence_length = _get_max_sequence_length(olmo_config, max_sequence_length)

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    loaded = load_model(os.path.join(input_base_path, "model_and_optim"))["model"]
    print(f"Loaded {len(loaded)} keys from checkpoint")

    full_state_dict: dict[str, torch.Tensor] = {}
    param_count = 0
    for layer_i in range(n_layers):
        layer_state = _convert_attention_layer_weights(loaded, layer_i)
        full_state_dict.update(layer_state)
        param_count += sum(v.numel() for v in layer_state.values())
        print(f"Converted layer {layer_i} (full_attention)")

    full_state_dict["model.embed_tokens.weight"] = loaded["embeddings.weight"]
    full_state_dict["model.embed_norm.weight"] = loaded["embedding_norm.weight"]
    full_state_dict["model.norm.weight"] = loaded["lm_head.norm.weight"]
    full_state_dict["lm_head.weight"] = loaded["lm_head.w_out.weight"]
    param_count += sum(
        loaded[key].numel()
        for key in ("embeddings.weight", "embedding_norm.weight", "lm_head.norm.weight", "lm_head.w_out.weight")
    )

    full_state_dict = {key: value.to(dtype) for key, value in full_state_dict.items()}
    print(f"Total parameters: {param_count}")

    config = OlmoHybridSmallConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=dim,
        intermediate_size=feed_forward_config["hidden_size"],
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=max_sequence_length,
        pad_token_id=tokenizer_config.get("pad_token_id"),
        bos_token_id=tokenizer_config.get("bos_token_id"),
        eos_token_id=tokenizer_config.get("eos_token_id"),
        tie_word_embeddings=False,
        rms_norm_eps=block_config.get("layer_norm", {}).get("eps", 1e-6),
        layer_types=["full_attention"] * n_layers,
    )
    config.rope_parameters = {"rope_theta": None}
    config.architectures = ["OlmoHybridSmallForCausalLM"]
    config.save_pretrained(model_path)

    safetensors_path = os.path.join(model_path, "model.safetensors")
    save_file(full_state_dict, safetensors_path)
    print(f"Saved weights to {safetensors_path}")

    del full_state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        _write_tokenizer(model_path, tokenizer_id, tokenizer_config, max_sequence_length)

    _patch_nope_config(model_path)
    print(f"Conversion complete. Model saved to {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Yashas OLMo transformer checkpoints to HF format.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of the checkpoint, which contains config.json and model_and_optim/.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not save the tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer identifier. Defaults to the one in the checkpoint config.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer.",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=None,
        help="Max sequence length. If not set, reads from config.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=list(DTYPE_MAP.keys()),
        help="Output dtype for model weights. Defaults to bfloat16.",
    )
    args = parser.parse_args()

    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        include_tokenizer=args.include_tokenizer,
        tokenizer_id=args.tokenizer,
        max_sequence_length=args.max_sequence_length,
        dtype=DTYPE_MAP[args.dtype],
    )


if __name__ == "__main__":
    main()
