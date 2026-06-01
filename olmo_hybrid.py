from __future__ import annotations

import argparse
import gc
import os
import shutil
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, OlmoHybridConfig, OlmoHybridForCausalLM

from baseline import load_model, write_json
from pure_gdn import (
    _conv_weight_for_hf,
    _get_max_sequence_length,
    _get_tokenizer_config,
    _gdn_required,
    _required,
    read_json,
)


def _write_tokenizer(output_path: Path, tokenizer_id: str) -> None:
    print(f"Saving a tokenizer to {output_path}.")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    tokenizer.save_pretrained(output_path)


def _layer_types(input_config: dict[str, Any]) -> list[str]:
    model_config = input_config["model"]
    block_config = model_config["block"]
    n_layers = model_config["n_layers"]

    sliding_window = block_config.get("attention", {}).get("sliding_window")
    if sliding_window is not None and "pattern" in sliding_window:
        pattern = sliding_window["pattern"]
        layer_types = [
            "full_attention" if pattern[layer_i % len(pattern)] == -1 else "linear_attention"
            for layer_i in range(n_layers)
        ]
        if sliding_window.get("force_full_attention_on_first_layer", False):
            layer_types[0] = "full_attention"
        if sliding_window.get("force_full_attention_on_last_layer", False):
            layer_types[-1] = "full_attention"
        return layer_types

    if "fla" in block_config or "sequence_mixer" in block_config:
        return ["linear_attention"] * n_layers

    return ["full_attention"] * n_layers


def _make_config(input_config: dict[str, Any]) -> OlmoHybridConfig:
    model_config = input_config["model"]
    block_config = model_config["block"]
    attention_config = block_config.get("attention", block_config.get("sequence_mixer", {}))
    feed_forward_config = block_config["feed_forward"]
    tokenizer_config = _get_tokenizer_config(input_config)

    n_layers = model_config["n_layers"]
    n_heads = attention_config["n_heads"]
    head_dim = model_config["d_model"] // n_heads
    rope = attention_config.get("rope") or {}

    config = OlmoHybridConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["d_model"],
        intermediate_size=feed_forward_config["hidden_size"],
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=attention_config.get("n_kv_heads", n_heads),
        hidden_act=feed_forward_config.get("activation", "silu"),
        max_position_embeddings=_get_max_sequence_length(input_config),
        initializer_range=model_config.get("init_std", 0.02),
        rms_norm_eps=block_config["layer_norm"]["eps"],
        pad_token_id=tokenizer_config["pad_token_id"],
        bos_token_id=None,
        eos_token_id=tokenizer_config["eos_token_id"],
        tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": rope["theta"]} if rope else None,
        attention_bias=attention_config.get("bias", False),
        layer_types=_layer_types(input_config),
        linear_num_key_heads=n_heads,
        linear_num_value_heads=n_heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=2 * head_dim,
        linear_allow_neg_eigval=True,
    )
    config.architectures = ["OlmoHybridForCausalLM"]
    return config


def _full_attention_layer_state_dict(loaded: dict[str, torch.Tensor], layer_i: int) -> dict[str, torch.Tensor]:
    prefix = f"blocks.{layer_i}"
    return {
        f"model.layers.{layer_i}.self_attn.q_proj.weight": _required(loaded, f"{prefix}.attention.w_q.weight"),
        f"model.layers.{layer_i}.self_attn.k_proj.weight": _required(loaded, f"{prefix}.attention.w_k.weight"),
        f"model.layers.{layer_i}.self_attn.v_proj.weight": _required(loaded, f"{prefix}.attention.w_v.weight"),
        f"model.layers.{layer_i}.self_attn.o_proj.weight": _required(loaded, f"{prefix}.attention.w_out.weight"),
        f"model.layers.{layer_i}.self_attn.q_norm.weight": _required(loaded, f"{prefix}.attention.q_norm.weight"),
        f"model.layers.{layer_i}.self_attn.k_norm.weight": _required(loaded, f"{prefix}.attention.k_norm.weight"),
        f"model.layers.{layer_i}.mlp.gate_proj.weight": _required(loaded, f"{prefix}.feed_forward.w1.weight"),
        f"model.layers.{layer_i}.mlp.down_proj.weight": _required(loaded, f"{prefix}.feed_forward.w2.weight"),
        f"model.layers.{layer_i}.mlp.up_proj.weight": _required(loaded, f"{prefix}.feed_forward.w3.weight"),
        f"model.layers.{layer_i}.post_attention_layernorm.weight": _required(
            loaded, f"{prefix}.attention_norm.weight"
        ),
        f"model.layers.{layer_i}.post_feedforward_layernorm.weight": _required(
            loaded, f"{prefix}.feed_forward_norm.weight"
        ),
    }


def _linear_attention_layer_state_dict(loaded: dict[str, torch.Tensor], layer_i: int) -> dict[str, torch.Tensor]:
    prefix = f"blocks.{layer_i}"
    return {
        f"model.layers.{layer_i}.mlp.gate_proj.weight": _required(loaded, f"{prefix}.feed_forward.w1.weight"),
        f"model.layers.{layer_i}.mlp.down_proj.weight": _required(loaded, f"{prefix}.feed_forward.w2.weight"),
        f"model.layers.{layer_i}.mlp.up_proj.weight": _required(loaded, f"{prefix}.feed_forward.w3.weight"),
        f"model.layers.{layer_i}.input_layernorm.weight": _required(loaded, f"{prefix}.attention_norm.weight"),
        f"model.layers.{layer_i}.post_attention_layernorm.weight": _required(
            loaded, f"{prefix}.feed_forward_norm.weight"
        ),
        f"model.layers.{layer_i}.linear_attn.A_log": _gdn_required(loaded, layer_i, "A_log"),
        f"model.layers.{layer_i}.linear_attn.dt_bias": _gdn_required(loaded, layer_i, "dt_bias"),
        f"model.layers.{layer_i}.linear_attn.q_proj.weight": _gdn_required(loaded, layer_i, "w_q.weight"),
        f"model.layers.{layer_i}.linear_attn.k_proj.weight": _gdn_required(loaded, layer_i, "w_k.weight"),
        f"model.layers.{layer_i}.linear_attn.v_proj.weight": _gdn_required(loaded, layer_i, "w_v.weight"),
        f"model.layers.{layer_i}.linear_attn.a_proj.weight": _gdn_required(loaded, layer_i, "w_a.weight"),
        f"model.layers.{layer_i}.linear_attn.b_proj.weight": _gdn_required(loaded, layer_i, "w_b.weight"),
        f"model.layers.{layer_i}.linear_attn.g_proj.weight": _gdn_required(loaded, layer_i, "w_g.weight"),
        f"model.layers.{layer_i}.linear_attn.o_proj.weight": _gdn_required(loaded, layer_i, "w_out.weight"),
        f"model.layers.{layer_i}.linear_attn.q_conv1d.weight": _conv_weight_for_hf(
            _gdn_required(loaded, layer_i, "q_conv1d.weight")
        ),
        f"model.layers.{layer_i}.linear_attn.k_conv1d.weight": _conv_weight_for_hf(
            _gdn_required(loaded, layer_i, "k_conv1d.weight")
        ),
        f"model.layers.{layer_i}.linear_attn.v_conv1d.weight": _conv_weight_for_hf(
            _gdn_required(loaded, layer_i, "v_conv1d.weight")
        ),
        f"model.layers.{layer_i}.linear_attn.o_norm.weight": _gdn_required(loaded, layer_i, "o_norm.weight"),
    }


def _layer_state_dict(
    loaded: dict[str, torch.Tensor],
    layer_i: int,
    layer_type: str,
) -> dict[str, torch.Tensor]:
    if layer_type == "full_attention":
        return _full_attention_layer_state_dict(loaded, layer_i)
    if layer_type == "linear_attention":
        return _linear_attention_layer_state_dict(loaded, layer_i)
    raise ValueError(f"Unsupported OlmoHybrid layer type: {layer_type}")


def write_model(
    model_path: str,
    input_base_path: str,
    include_tokenizer: bool = True,
    tokenizer_id: str | Path | None = None,
    tmp_cleanup: bool = True,
) -> None:
    output_path = Path(model_path)
    output_path.mkdir(parents=True, exist_ok=True)
    tmp_model_path = output_path / "tmp"
    tmp_model_path.mkdir(parents=True, exist_ok=True)

    input_config = read_json(Path(input_base_path) / "config.json")
    model_config = input_config["model"]
    tokenizer_config = _get_tokenizer_config(input_config)
    layer_types = _layer_types(input_config)
    n_layers = model_config["n_layers"]

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    loaded = load_model(os.path.join(input_base_path, "model_and_optim"))["model"]

    param_count = 0
    index_dict: dict[str, Any] = {"weight_map": {}}
    for layer_i, layer_type in enumerate(layer_types):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        state_dict = _layer_state_dict(loaded, layer_i, layer_type)

        for key, tensor in state_dict.items():
            index_dict["weight_map"][key] = filename
            param_count += tensor.numel()
        torch.save(state_dict, tmp_model_path / filename)

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    state_dict = {
        "model.embed_tokens.weight": _required(loaded, "embeddings.weight"),
        "model.norm.weight": _required(loaded, "lm_head.norm.weight"),
        "lm_head.weight": _required(loaded, "lm_head.w_out.weight"),
    }

    for key, tensor in state_dict.items():
        index_dict["weight_map"][key] = filename
        param_count += tensor.numel()
    torch.save(state_dict, tmp_model_path / filename)

    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, tmp_model_path / "pytorch_model.bin.index.json")

    config = _make_config(input_config)
    config.save_pretrained(tmp_model_path)

    del state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        tokenizer_id = tokenizer_id or tokenizer_config["identifier"]
        _write_tokenizer(output_path, str(tokenizer_id))

    print("Loading the checkpoint in an OlmoHybrid model.")
    model = OlmoHybridForCausalLM.from_pretrained(tmp_model_path, dtype=torch.bfloat16)
    print("Resizing token embeddings to match tokenizer config.")
    model.resize_token_embeddings(tokenizer_config["vocab_size"])
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(output_path)

    if tmp_cleanup:
        shutil.rmtree(tmp_model_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo Core weights, containing config.json and model_and_optim.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not save the tokenizer.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Tokenizer id/path. Defaults to the tokenizer identifier in the config.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer.",
    )
    parser.add_argument(
        "--no_tmp_cleanup",
        action="store_false",
        dest="tmp_cleanup",
        help="If passed, don't remove temp dir at end of HF conversion.",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        include_tokenizer=args.include_tokenizer,
        tokenizer_id=args.tokenizer,
        tmp_cleanup=args.tmp_cleanup,
    )


if __name__ == "__main__":
    main()
