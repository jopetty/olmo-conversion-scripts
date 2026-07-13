"""Convert a synthetic reordered-norm transformer checkpoint to Hugging Face.

The synthetic transformer uses OLMo-core's reordered-norm block and a
per-layer sliding-window schedule. The emitted custom model keeps those
semantics while reusing Hugging Face's OLMo Hybrid attention implementation.
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


class SyntheticTransformerConfig(OlmoHybridConfig):
    model_type = "synthetic_transformer"

    def validate_architecture(self):
        # OlmoHybrid normally requires at least one GDN layer. Synthetic
        # transformers deliberately contain only quadratic-attention layers.
        return
'''


MODELING_CODE = '''import torch

from transformers.masking_utils import create_causal_mask, sliding_window_overlay
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.olmo_hybrid.modeling_olmo_hybrid import (
    OlmoHybridForCausalLM,
    OlmoHybridModel,
)

from .configuration_synthetic_transformer import SyntheticTransformerConfig


class SyntheticTransformerModel(OlmoHybridModel):
    config_class = SyntheticTransformerConfig

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if use_cache or past_key_values is not None:
            raise ValueError("SyntheticTransformerModel does not support KV caching.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)

        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        for layer_idx, decoder_layer in enumerate(self.layers):
            window = self.config.sliding_window_pattern[layer_idx % len(self.config.sliding_window_pattern)]
            layer_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                position_ids=position_ids,
                and_mask_function=sliding_window_overlay(window) if window != -1 else None,
            )
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
                **kwargs,
            )

        return BaseModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


class SyntheticTransformerForCausalLM(OlmoHybridForCausalLM):
    config_class = SyntheticTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = SyntheticTransformerModel(config)
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
    if isinstance(config.get("dataset"), dict) and isinstance(config["dataset"].get("tokenizer"), dict):
        return config["dataset"]["tokenizer"]
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
    value = config.get("dataset", {}).get("sequence_length")
    if isinstance(value, int):
        return value
    lengths = [length for length in _find_values(config.get("instance_sources", []), "sequence_length") if isinstance(length, int)]
    return max(lengths) if lengths else 8192


def _required(loaded: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return loaded[key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint is missing required tensor {key!r}.") from exc


def _validate_config(model: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    block = model["block"]
    if block.get("name") != "reordered_norm":
        raise ValueError("transformer_synthetic.py requires a reordered_norm synthetic transformer block.")
    attention = block["sequence_mixer"]
    if attention.get("type") != "attention":
        raise ValueError("Synthetic transformer sequence mixer must be full attention.")
    if attention.get("use_head_qk_norm", False):
        raise ValueError("Synthetic transformer requires whole-vector QK norms.")
    rope = attention.get("rope")
    if not rope or rope.get("name") != "default":
        raise ValueError("Synthetic transformer requires default RoPE.")
    pattern = attention.get("sliding_window", {}).get("pattern")
    if not pattern or not all(isinstance(window, int) and (window > 0 or window == -1) for window in pattern):
        raise ValueError("Synthetic transformer requires a valid sliding-window pattern.")
    if model.get("embedding_norm") is not None or model.get("embed_scale") is not None:
        raise ValueError("Synthetic reordered-norm transformer must not use embedding_norm or embed_scale.")
    if attention.get("gate") is not None:
        raise ValueError("Synthetic reordered-norm transformer must not use an attention gate.")
    return block, attention


def _layer_state(loaded: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
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


def _write_custom_code(output_dir: Path) -> None:
    (output_dir / "configuration_synthetic_transformer.py").write_text(CONFIGURATION_CODE)
    (output_dir / "modeling_synthetic_transformer.py").write_text(MODELING_CODE)


def _write_config(
    output_dir: Path,
    model: dict[str, Any],
    block: dict[str, Any],
    attention: dict[str, Any],
    tokenizer: dict[str, Any],
    max_sequence_length: int,
) -> None:
    rope = attention["rope"]
    pattern = attention["sliding_window"]["pattern"]
    hidden_size = model["d_model"]
    num_heads = attention["n_heads"]
    config = {
        "architectures": ["SyntheticTransformerForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_synthetic_transformer.SyntheticTransformerConfig",
            "AutoModel": "modeling_synthetic_transformer.SyntheticTransformerModel",
            "AutoModelForCausalLM": "modeling_synthetic_transformer.SyntheticTransformerForCausalLM",
        },
        "model_type": "synthetic_transformer",
        # Preserve the padded checkpoint vocabulary; the tokenizer has fewer usable
        # tokens, but resizing would delete valid LM-head rows and change logits.
        "vocab_size": model["vocab_size"],
        "hidden_size": hidden_size,
        "intermediate_size": block["feed_forward"]["hidden_size"],
        "num_hidden_layers": model["n_layers"],
        "num_attention_heads": num_heads,
        "num_key_value_heads": attention.get("n_kv_heads", num_heads),
        "head_dim": attention.get("head_dim", hidden_size // num_heads),
        "hidden_act": block["feed_forward"].get("activation", "silu"),
        "max_position_embeddings": max_sequence_length,
        "rms_norm_eps": block["layer_norm"].get("eps", 1e-6),
        "use_cache": False,
        "rope_parameters": {"rope_type": "default", "rope_theta": rope["theta"]},
        "sliding_window_pattern": pattern,
        "layer_types": ["full_attention"] * model["n_layers"],
        "attention_bias": attention.get("bias", False),
        "attention_dropout": attention.get("dropout", 0.0),
        "pad_token_id": tokenizer.get("pad_token_id"),
        "bos_token_id": tokenizer.get("bos_token_id"),
        "eos_token_id": tokenizer.get("eos_token_id"),
        "tie_word_embeddings": model.get("tie_word_embeddings", False),
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
    block, attention = _validate_config(model)
    tokenizer = _tokenizer_config(experiment)
    max_sequence_length = _max_sequence_length(experiment, max_sequence_length)

    print(f"Loading checkpoint from {source_dir}")
    loaded = load_model(str(source_dir / "model_and_optim"))["model"]
    state: dict[str, torch.Tensor] = {}
    for layer in range(model["n_layers"]):
        state.update(_layer_state(loaded, layer))
    state.update(
        {
            "model.embed_tokens.weight": _required(loaded, "embeddings.weight"),
            "model.norm.weight": _required(loaded, "lm_head.norm.weight"),
            "lm_head.weight": _required(loaded, "lm_head.w_out.weight"),
        }
    )
    state = {name: tensor.to(dtype) for name, tensor in state.items()}

    _write_custom_code(target_dir)
    _write_config(target_dir, model, block, attention, tokenizer, max_sequence_length)
    save_file(state, target_dir / "model.safetensors")
    if include_tokenizer:
        _write_tokenizer(target_dir, tokenizer_id, tokenizer, max_sequence_length)
    print(f"Converted synthetic transformer checkpoint to {target_dir}")

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
