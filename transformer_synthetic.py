"""Convert a synthetic peri-norm transformer checkpoint to Hugging Face format.

This converter deliberately does not use ``Olmo3ForCausalLM``.  Synthetic
transformer checkpoints use the OLMo-core peri-norm block, elementwise
attention gates, and scaled/normed embeddings, none of which are represented
by the stock OLMo 3 implementation.

The emitted checkpoint contains the small amount of custom modeling code
needed by ``AutoModelForCausalLM(..., trust_remote_code=True)``.
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


CONFIGURATION_CODE = '''from __future__ import annotations

from transformers import PreTrainedConfig


class SyntheticTransformerConfig(PreTrainedConfig):
    model_type = "synthetic_transformer"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=100352,
        hidden_size=128,
        intermediate_size=1024,
        num_hidden_layers=14,
        num_attention_heads=2,
        num_key_value_heads=None,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=100277,
        bos_token_id=None,
        eos_token_id=100257,
        tie_word_embeddings=False,
        embed_scale=None,
        embedding_norm_eps=1e-6,
        use_attention_gate=True,
        use_head_qk_norm=True,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.embed_scale = embed_scale
        self.embedding_norm_eps = embedding_norm_eps
        self.use_attention_gate = use_attention_gate
        self.use_head_qk_norm = use_head_qk_norm
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # OlmoHybridSmall uses this value to disable RoPE.
        self.rope_parameters = {"rope_theta": None}
        self.layer_types = ["full_attention"] * num_hidden_layers
'''


MODELING_CODE = '''from transformers.models.olmo_hybrid_small.modeling_olmo_hybrid_small import (
    OlmoHybridSmallForCausalLM,
    OlmoHybridSmallModel,
)

from .configuration_synthetic_transformer import SyntheticTransformerConfig


class SyntheticTransformerModel(OlmoHybridSmallModel):
    config_class = SyntheticTransformerConfig


class SyntheticTransformerForCausalLM(OlmoHybridSmallForCausalLM):
    config_class = SyntheticTransformerConfig
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
    if lengths:
        return max(lengths)
    return 8192


def _required(loaded: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return loaded[key]
    except KeyError as exc:
        raise KeyError(f"Checkpoint is missing required tensor {key!r}.") from exc


def _validate_config(model_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    block = model_config["block"]
    if block.get("name") != "peri_norm":
        raise ValueError("transformer_synthetic.py only supports a peri_norm synthetic transformer block.")
    attention = block["sequence_mixer"]
    if attention.get("type") != "attention":
        raise ValueError("Synthetic transformer sequence mixer must be full attention.")
    if attention.get("rope") is not None:
        raise ValueError("Synthetic transformer converter currently supports the NoPE synthetic configuration only.")
    gate = attention.get("gate")
    if not gate or gate.get("granularity") != "elementwise":
        raise ValueError("Synthetic transformer requires an elementwise attention gate.")
    if "embedding_norm" not in model_config or model_config.get("embed_scale") is None:
        raise ValueError("Synthetic transformer requires embedding_norm and embed_scale.")
    return block, attention


def _layer_state(loaded: dict[str, torch.Tensor], layer: int) -> dict[str, torch.Tensor]:
    source = f"blocks.{layer}"
    target = f"model.layers.{layer}"
    return {
        f"{target}.self_attn.q_proj.weight": _required(loaded, f"{source}.attention.w_q.weight"),
        f"{target}.self_attn.k_proj.weight": _required(loaded, f"{source}.attention.w_k.weight"),
        f"{target}.self_attn.v_proj.weight": _required(loaded, f"{source}.attention.w_v.weight"),
        f"{target}.self_attn.o_proj.weight": _required(loaded, f"{source}.attention.w_out.weight"),
        f"{target}.self_attn.attn_gate.weight": _required(loaded, f"{source}.attention.w_g.weight"),
        f"{target}.self_attn.q_norm.weight": _required(loaded, f"{source}.attention.q_norm.weight"),
        f"{target}.self_attn.k_norm.weight": _required(loaded, f"{source}.attention.k_norm.weight"),
        f"{target}.mlp.gate_proj.weight": _required(loaded, f"{source}.feed_forward.w1.weight"),
        f"{target}.mlp.down_proj.weight": _required(loaded, f"{source}.feed_forward.w2.weight"),
        f"{target}.mlp.up_proj.weight": _required(loaded, f"{source}.feed_forward.w3.weight"),
        f"{target}.input_layernorm.weight": _required(loaded, f"{source}.attention_norm.weight"),
        f"{target}.post_attention_layernorm.weight": _required(loaded, f"{source}.post_attention_norm.weight"),
        f"{target}.ffn_layernorm.weight": _required(loaded, f"{source}.feed_forward_norm.weight"),
        f"{target}.post_feedforward_layernorm.weight": _required(
            loaded, f"{source}.post_feed_forward_norm.weight"
        ),
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
        # Keep the checkpoint's padded vocabulary.  The tokenizer has fewer usable
        # tokens, but resizing would delete valid output rows and change logits.
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
        # The upstream hybrid-small cache requires at least one linear-attention
        # layer. Synthetic transformers are all quadratic-attention layers.
        "use_cache": False,
        "embedding_norm_eps": model["embedding_norm"].get("eps", 1e-6),
        "embed_scale": model["embed_scale"],
        "use_attention_gate": True,
        "use_head_qk_norm": attention.get("use_head_qk_norm", True),
        "attention_bias": attention.get("bias", False),
        "attention_dropout": attention.get("dropout", 0.0),
        "pad_token_id": tokenizer.get("pad_token_id"),
        "bos_token_id": tokenizer.get("bos_token_id"),
        "eos_token_id": tokenizer.get("eos_token_id"),
        "tie_word_embeddings": False,
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
            "model.embed_norm.weight": _required(loaded, "embedding_norm.weight"),
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
