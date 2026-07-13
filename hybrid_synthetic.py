"""Convert a synthetic reordered-norm GDN/attention hybrid checkpoint.

The stock HF ``OlmoHybrid`` linear-attention layer is pre-norm.  Synthetic
hybrid checkpoints use OLMo-core's reordered-norm block instead, so this
converter emits a small custom model which implements the matching GDN block.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, cast

import torch
from safetensors.torch import save_file
from transformers import AutoTokenizer

from scripts.hybrid import DTYPE_MAP, load_model


CONFIGURATION_CODE = '''from transformers import OlmoHybridConfig


class SyntheticHybridConfig(OlmoHybridConfig):
    model_type = "synthetic_hybrid"
'''


MODELING_CODE = '''import torch.nn as nn

from transformers.models.olmo_hybrid.modeling_olmo_hybrid import (
    OlmoHybridForCausalLM,
    OlmoHybridGatedDeltaNet,
    OlmoHybridMLP,
    OlmoHybridModel,
    OlmoHybridRMSNorm,
)

from .configuration_synthetic_hybrid import SyntheticHybridConfig


class SyntheticReorderedLinearAttentionDecoderLayer(nn.Module):
    """OLMo-core ReorderedNormTransformerBlock with a GatedDeltaNet mixer."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.linear_attn = OlmoHybridGatedDeltaNet(config, layer_idx)
        self.mlp = OlmoHybridMLP(config)
        self.post_attention_layernorm = OlmoHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = OlmoHybridRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_values=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.linear_attn(
            hidden_states=hidden_states,
            cache_params=past_key_values,
            attention_mask=attention_mask,
        )
        hidden_states = residual + self.post_attention_layernorm(hidden_states)

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        return residual + self.post_feedforward_layernorm(hidden_states)


class SyntheticHybridModel(OlmoHybridModel):
    config_class = SyntheticHybridConfig

    def __init__(self, config):
        super().__init__(config)
        for layer_idx, layer_type in enumerate(config.layer_types):
            if layer_type == "linear_attention":
                self.layers[layer_idx] = SyntheticReorderedLinearAttentionDecoderLayer(config, layer_idx)


class SyntheticHybridForCausalLM(OlmoHybridForCausalLM):
    config_class = SyntheticHybridConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SyntheticHybridModel(config)
        # The checkpoint loader supplies every parameter after this initialization.
        self.post_init()
'''


def _find_values(value: Any, key: str) -> list[Any]:
    if isinstance(value, dict):
        found = [value[key]] if key in value else []
        for child in value.values():
            found.extend(_find_values(child, key))
        return found
    if isinstance(value, list):
        found: list[Any] = []
        for child in value:
            found.extend(_find_values(child, key))
        return found
    return []


def _tokenizer_config(config: dict[str, Any]) -> dict[str, Any]:
    tokenizers = _find_values(config.get("instance_sources", []), "tokenizer")
    for tokenizer in tokenizers:
        if isinstance(tokenizer, dict):
            return tokenizer
    raise KeyError("Could not find a tokenizer configuration in the checkpoint config.")


def _max_sequence_length(config: dict[str, Any], override: int | None) -> int:
    if override is not None:
        return override
    value = config.get("train_module", {}).get("max_sequence_length")
    if isinstance(value, int):
        return value
    lengths = [length for length in _find_values(config.get("instance_sources", []), "sequence_length") if isinstance(length, int)]
    return max(lengths) if lengths else 8192


def _required(loaded: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return loaded[key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint is missing required tensor {key!r}.") from exc


def _layer_types(model: dict[str, Any]) -> list[str]:
    pattern = model.get("block_pattern")
    blocks = model["block"]
    if not pattern:
        raise ValueError("Synthetic hybrid converter requires a named block_pattern.")
    layer_types: list[str] = []
    for layer in range(model["n_layers"]):
        block_name = pattern[layer % len(pattern)]
        mixer = blocks[block_name].get("sequence_mixer", {})
        mixer_type = mixer.get("type")
        if mixer_type == "gated_delta_net":
            layer_types.append("linear_attention")
        elif mixer_type == "attention":
            layer_types.append("full_attention")
        else:
            raise ValueError(f"Unsupported synthetic hybrid mixer {mixer_type!r} in block {block_name!r}.")
    return layer_types


def _validate_config(model: dict[str, Any], layer_types: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    blocks = model["block"]
    gdn_block = next(block for block in blocks.values() if block.get("sequence_mixer", {}).get("type") == "gated_delta_net")
    attn_block = next(block for block in blocks.values() if block.get("sequence_mixer", {}).get("type") == "attention")
    if gdn_block.get("name") != "reordered_norm" or attn_block.get("name") != "reordered_norm":
        raise ValueError("Synthetic hybrid converter requires reordered_norm GDN and attention blocks.")
    attention = attn_block["sequence_mixer"]
    if attention.get("use_head_qk_norm", False):
        raise ValueError("The synthetic hybrid HF architecture currently requires whole-vector QK norms.")
    window = attention.get("sliding_window", {}).get("pattern")
    if window:
        for layer, layer_type in enumerate(layer_types):
            if layer_type == "full_attention" and window[layer % len(window)] != -1:
                raise ValueError("HF synthetic hybrid only supports full, not sliding-window, quadratic-attention layers.")
    return gdn_block, attn_block


def _gdn_weight(loaded: dict[str, torch.Tensor], layer: int, suffix: str) -> torch.Tensor:
    return _required(loaded, f"blocks.{layer}.attention.{suffix}")


def _attention_state(loaded: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
    source = f"blocks.{layer}"
    target = f"model.layers.{layer}"
    return {
        f"{target}.self_attn.q_proj.weight": _required(loaded, f"{source}.attention.w_q.weight"),
        f"{target}.self_attn.k_proj.weight": _required(loaded, f"{source}.attention.w_k.weight"),
        f"{target}.self_attn.v_proj.weight": _required(loaded, f"{source}.attention.w_v.weight"),
        f"{target}.self_attn.o_proj.weight": _required(loaded, f"{source}.attention.w_out.weight"),
        f"{target}.self_attn.q_norm.weight": _required(loaded, f"{source}.attention.q_norm.weight"),
        f"{target}.self_attn.k_norm.weight": _required(loaded, f"{source}.attention.k_norm.weight"),
        f"{target}.mlp.gate_proj.weight": _required(loaded, f"{source}.feed_forward.w1.weight"),
        f"{target}.mlp.down_proj.weight": _required(loaded, f"{source}.feed_forward.w2.weight"),
        f"{target}.mlp.up_proj.weight": _required(loaded, f"{source}.feed_forward.w3.weight"),
        f"{target}.post_attention_layernorm.weight": _required(loaded, f"{source}.attention_norm.weight"),
        f"{target}.post_feedforward_layernorm.weight": _required(loaded, f"{source}.feed_forward_norm.weight"),
    }


def _gdn_state(loaded: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
    source = f"blocks.{layer}"
    target = f"model.layers.{layer}"
    def conv(name: str) -> torch.Tensor:
        tensor = _gdn_weight(loaded, layer, f"{name}.weight")
        return tensor.unsqueeze(1) if tensor.ndim == 2 else tensor

    return {
        f"{target}.linear_attn.q_proj.weight": _gdn_weight(loaded, layer, "w_q.weight"),
        f"{target}.linear_attn.k_proj.weight": _gdn_weight(loaded, layer, "w_k.weight"),
        f"{target}.linear_attn.v_proj.weight": _gdn_weight(loaded, layer, "w_v.weight"),
        f"{target}.linear_attn.o_proj.weight": _gdn_weight(loaded, layer, "w_out.weight"),
        f"{target}.linear_attn.g_proj.weight": _gdn_weight(loaded, layer, "w_g.weight"),
        f"{target}.linear_attn.a_proj.weight": _gdn_weight(loaded, layer, "w_a.weight"),
        f"{target}.linear_attn.b_proj.weight": _gdn_weight(loaded, layer, "w_b.weight"),
        f"{target}.linear_attn.o_norm.weight": _gdn_weight(loaded, layer, "o_norm.weight"),
        f"{target}.linear_attn.q_conv1d.weight": conv("q_conv1d"),
        f"{target}.linear_attn.k_conv1d.weight": conv("k_conv1d"),
        f"{target}.linear_attn.v_conv1d.weight": conv("v_conv1d"),
        f"{target}.linear_attn.A_log": _gdn_weight(loaded, layer, "A_log"),
        f"{target}.linear_attn.dt_bias": _gdn_weight(loaded, layer, "dt_bias"),
        f"{target}.mlp.gate_proj.weight": _required(loaded, f"{source}.feed_forward.w1.weight"),
        f"{target}.mlp.down_proj.weight": _required(loaded, f"{source}.feed_forward.w2.weight"),
        f"{target}.mlp.up_proj.weight": _required(loaded, f"{source}.feed_forward.w3.weight"),
        # Reordered norm: both norms are applied to sub-block outputs, never inputs.
        f"{target}.post_attention_layernorm.weight": _required(loaded, f"{source}.attention_norm.weight"),
        f"{target}.post_feedforward_layernorm.weight": _required(loaded, f"{source}.feed_forward_norm.weight"),
    }


def _write_custom_code(output_dir: Path) -> None:
    (output_dir / "configuration_synthetic_hybrid.py").write_text(CONFIGURATION_CODE)
    (output_dir / "modeling_synthetic_hybrid.py").write_text(MODELING_CODE)


def _write_config(
    output_dir: Path,
    model: dict[str, Any],
    gdn_block: dict[str, Any],
    attn_block: dict[str, Any],
    tokenizer: dict[str, Any],
    layer_types: list[str],
    max_sequence_length: int,
) -> None:
    attention = attn_block["sequence_mixer"]
    gdn = gdn_block["sequence_mixer"]
    hidden_size = model["d_model"]
    num_heads = attention["n_heads"]
    rope = attention.get("rope")
    config = {
        "architectures": ["SyntheticHybridForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_synthetic_hybrid.SyntheticHybridConfig",
            "AutoModel": "modeling_synthetic_hybrid.SyntheticHybridModel",
            "AutoModelForCausalLM": "modeling_synthetic_hybrid.SyntheticHybridForCausalLM",
        },
        "model_type": "synthetic_hybrid",
        "vocab_size": model["vocab_size"],
        "hidden_size": hidden_size,
        "intermediate_size": attn_block["feed_forward"]["hidden_size"],
        "num_hidden_layers": model["n_layers"],
        "num_attention_heads": num_heads,
        "num_key_value_heads": attention.get("n_kv_heads", num_heads),
        "hidden_act": attn_block["feed_forward"].get("activation", "silu"),
        "max_position_embeddings": max_sequence_length,
        "rms_norm_eps": attn_block["layer_norm"].get("eps", 1e-6),
        "pad_token_id": tokenizer.get("pad_token_id"),
        "bos_token_id": tokenizer.get("bos_token_id"),
        "eos_token_id": tokenizer.get("eos_token_id"),
        "tie_word_embeddings": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": rope["theta"]} if rope else None,
        "attention_bias": attention.get("bias", False),
        "layer_types": layer_types,
        "linear_num_key_heads": gdn["n_heads"],
        "linear_num_value_heads": gdn.get("n_v_heads", gdn["n_heads"]),
        "linear_key_head_dim": gdn["head_dim"],
        "linear_value_head_dim": int(gdn["head_dim"] * gdn.get("expand_v", 2.0)),
        "linear_conv_kernel_dim": gdn.get("conv_size", 4),
        "linear_allow_neg_eigval": gdn.get("allow_neg_eigval", True),
        "torch_dtype": "bfloat16",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def _write_tokenizer(output_dir: Path, tokenizer_id: str | None, tokenizer: dict[str, Any], max_sequence_length: int) -> None:
    tokenizer_id = tokenizer_id or tokenizer.get("identifier")
    if not tokenizer_id:
        print("No tokenizer identifier in checkpoint config; not writing tokenizer files.")
        return
    hf_tokenizer = cast(Any, AutoTokenizer.from_pretrained(tokenizer_id))
    hf_tokenizer.model_max_length = max_sequence_length
    hf_tokenizer.pad_token_id = tokenizer.get("pad_token_id")
    hf_tokenizer.bos_token_id = tokenizer.get("bos_token_id")
    hf_tokenizer.eos_token_id = tokenizer.get("eos_token_id")
    hf_tokenizer.save_pretrained(output_dir)


def write_model(
    input_dir: str,
    output_dir: str,
    *,
    include_tokenizer: bool = True,
    tokenizer_id: str | None = None,
    max_sequence_length: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    source_dir = Path(input_dir)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    experiment = json.loads((source_dir / "config.json").read_text())
    model = experiment["model"]
    layer_types = _layer_types(model)
    gdn_block, attn_block = _validate_config(model, layer_types)
    tokenizer = _tokenizer_config(experiment)
    max_sequence_length = _max_sequence_length(experiment, max_sequence_length)

    print(f"Loading checkpoint from {source_dir}")
    loaded = load_model(str(source_dir / "model_and_optim"))["model"]
    state: dict[str, torch.Tensor] = {}
    for layer, layer_type in enumerate(layer_types):
        state.update(_gdn_state(loaded, layer) if layer_type == "linear_attention" else _attention_state(loaded, layer))
    state.update(
        {
            "model.embed_tokens.weight": _required(loaded, "embeddings.weight"),
            "model.norm.weight": _required(loaded, "lm_head.norm.weight"),
            "lm_head.weight": _required(loaded, "lm_head.w_out.weight"),
        }
    )
    state = {name: tensor.to(dtype) for name, tensor in state.items()}

    _write_custom_code(target_dir)
    _write_config(target_dir, model, gdn_block, attn_block, tokenizer, layer_types, max_sequence_length)
    save_file(state, target_dir / "model.safetensors")
    if include_tokenizer:
        _write_tokenizer(target_dir, tokenizer_id, tokenizer, max_sequence_length)
    print(f"Converted synthetic hybrid checkpoint to {target_dir}")

    del state
    del loaded
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--max_sequence_length", type=int, default=None)
    parser.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="bfloat16")
    parser.add_argument("--no_tokenizer", action="store_false", dest="include_tokenizer")
    args = parser.parse_args()
    write_model(
        args.input_dir,
        args.output_dir,
        include_tokenizer=args.include_tokenizer,
        tokenizer_id=args.tokenizer,
        max_sequence_length=args.max_sequence_length,
        dtype=DTYPE_MAP[args.dtype],
    )


if __name__ == "__main__":
    main()
