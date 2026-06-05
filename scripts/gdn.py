from __future__ import annotations

import argparse
import gc
import io
import json
import os
from os import PathLike
import pickle
import shutil
import traceback
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed.checkpoint.metadata import Metadata, MetadataIndex, StorageMeta
from torch.distributed.checkpoint.planner import LoadItemType, ReadItem
from torch.futures import Future

from transformers import AutoModelForCausalLM, AutoTokenizer, PretrainedConfig
from transformers.utils import strtobool


class PureGDNConfig(PretrainedConfig):
    model_type = "pure_gdn"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size: int = 100352,
        hidden_size: int = 2048,
        intermediate_size: int = 8192,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 16,
        num_key_value_heads: int | None = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        pad_token_id: int | None = 100277,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = 100257,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        layer_types: list[str] | None = None,
        linear_num_key_heads: int | None = None,
        linear_num_value_heads: int | None = None,
        linear_key_head_dim: int | None = None,
        linear_value_head_dim: int | None = None,
        linear_a_log_min: float = 0.0,
        linear_a_log_max: float = 16.0,
        linear_dt_min: float = 0.001,
        linear_dt_max: float = 0.1,
        linear_dt_init_floor: float = 1e-4,
        linear_conv_kernel_dim: int = 4,
        linear_allow_neg_eigval: bool = True,
        **kwargs: Any,
    ) -> None:
        config_kwargs = {
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "tie_word_embeddings": tie_word_embeddings,
            **kwargs,
        }
        cast(Any, super()).__init__(**config_kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types or ["linear_attention"] * num_hidden_layers
        self.linear_num_key_heads = linear_num_key_heads or num_attention_heads
        self.linear_num_value_heads = linear_num_value_heads or num_attention_heads
        self.linear_key_head_dim = linear_key_head_dim or hidden_size // self.linear_num_key_heads
        self.linear_value_head_dim = linear_value_head_dim or 2 * self.linear_key_head_dim
        self.linear_a_log_min = linear_a_log_min
        self.linear_a_log_max = linear_a_log_max
        self.linear_dt_min = linear_dt_min
        self.linear_dt_max = linear_dt_max
        self.linear_dt_init_floor = linear_dt_init_floor
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_allow_neg_eigval = linear_allow_neg_eigval

    def validate_architecture(self) -> None:
        if "linear_attention" not in self.layer_types:
            raise ValueError("PureGDN expects at least one 'linear_attention' layer.")


CONFIGURATION_PURE_GDN = '''\
from transformers import PretrainedConfig


class PureGDNConfig(PretrainedConfig):
    model_type = "pure_gdn"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        *,
        vocab_size=100352,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=8192,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=100277,
        bos_token_id=None,
        eos_token_id=100257,
        tie_word_embeddings=False,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        layer_types=None,
        linear_num_key_heads=None,
        linear_num_value_heads=None,
        linear_key_head_dim=None,
        linear_value_head_dim=None,
        linear_a_log_min=0.0,
        linear_a_log_max=16.0,
        linear_dt_min=0.001,
        linear_dt_max=0.1,
        linear_dt_init_floor=1e-4,
        linear_conv_kernel_dim=4,
        linear_allow_neg_eigval=True,
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
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.layer_types = layer_types or ["linear_attention"] * num_hidden_layers
        self.linear_num_key_heads = linear_num_key_heads or num_attention_heads
        self.linear_num_value_heads = linear_num_value_heads or num_attention_heads
        self.linear_key_head_dim = linear_key_head_dim or hidden_size // self.linear_num_key_heads
        self.linear_value_head_dim = linear_value_head_dim or 2 * self.linear_key_head_dim
        self.linear_a_log_min = linear_a_log_min
        self.linear_a_log_max = linear_a_log_max
        self.linear_dt_min = linear_dt_min
        self.linear_dt_max = linear_dt_max
        self.linear_dt_init_floor = linear_dt_init_floor
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_allow_neg_eigval = linear_allow_neg_eigval

    def validate_architecture(self):
        if "linear_attention" not in self.layer_types:
            raise ValueError("PureGDN expects at least one 'linear_attention' layer.")
'''


MODELING_PURE_GDN = '''\
from transformers.models.olmo_hybrid.modeling_olmo_hybrid import (
    OlmoHybridForCausalLM,
    OlmoHybridModel,
    OlmoHybridPreTrainedModel,
)

from .configuration_pure_gdn import PureGDNConfig


class PureGDNPreTrainedModel(OlmoHybridPreTrainedModel):
    config_class = PureGDNConfig


class PureGDNModel(OlmoHybridModel):
    config_class = PureGDNConfig


class PureGDNForCausalLM(OlmoHybridForCausalLM):
    config_class = PureGDNConfig
'''


def read_json(path: Path | str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(text: dict[str, Any], path: Path | str) -> None:
    with open(path, "w") as f:
        json.dump(text, f)


def normalize_path(path: Path | str | PathLike[Any]) -> str:
    return str(path).rstrip("/").replace("file://", "")


def get_bytes_range(path: Path | str, bytes_start: int, num_bytes: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(bytes_start)
        return f.read(num_bytes)


def _narrow_tensor_by_index(tensor: torch.Tensor, offsets: Sequence[int], sizes: Sequence[int]) -> torch.Tensor:
    narrowed_tensor = tensor
    for idx, (offset, size) in enumerate(zip(offsets, sizes)):
        if size < tensor.size(idx):
            narrowed_tensor = narrowed_tensor.narrow(idx, offset, size)
    return narrowed_tensor


@dataclass
class _StorageInfo:
    relative_path: str
    offset: int
    length: int


class RemoteFileSystemReader(dist_cp.StorageReader):
    def __init__(self, path: Path | str, *, thread_count: int | None = None) -> None:
        super().__init__()
        if thread_count is not None and thread_count <= 0:
            raise ValueError("thread count must be at least 1")
        self.path = normalize_path(path)
        self.thread_count = thread_count or 1
        self.storage_data: dict[MetadataIndex, _StorageInfo] = {}
        self.load_id = str(uuid.uuid4())
        self._metadata: Metadata | None = None

    def _get_content_for_read(self, read_item: ReadItem) -> tuple[ReadItem, bytes]:
        sinfo = self.storage_data[read_item.storage_index]
        content = get_bytes_range(f"{self.path}/{sinfo.relative_path}", sinfo.offset, sinfo.length)
        return (read_item, content)

    def reset(self, checkpoint_id: str | PathLike[Any] | None = None) -> None:
        self.storage_data = {}
        if checkpoint_id:
            self.path = normalize_path(checkpoint_id)
        self.load_id = str(uuid.uuid4())

    def read_data(self, plan: dist_cp.LoadPlan, planner: dist_cp.LoadPlanner) -> Future[None]:
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = [executor.submit(self._get_content_for_read, read_item) for read_item in plan.items]
            results = []
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except BaseException as exc:
                    raise RuntimeError(f"Original error:\n{traceback.format_exc()}") from exc

        for read_item, content in results:
            bytes_io = io.BytesIO(content)
            bytes_io.seek(0)
            if read_item.type == LoadItemType.BYTE_IO:
                planner.load_bytes(read_item, bytes_io)
            else:
                tensor = cast(torch.Tensor, torch.load(bytes_io, map_location="cpu", weights_only=False))
                tensor = _narrow_tensor_by_index(tensor, read_item.storage_offsets, read_item.lengths)
                target_tensor = planner.resolve_tensor(read_item).detach()
                if target_tensor.size() != tensor.size():
                    raise RuntimeError(
                        f"Checkpoint tensor {read_item.storage_index} has size {tensor.size()}, "
                        f"expected {target_tensor.size()}"
                    )
                target_tensor.copy_(tensor)
                planner.commit_tensor(read_item, target_tensor)

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        if self._metadata is None:
            try:
                if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
                    raise ValueError(
                        "This checkpoint loader uses pickle. Set TRUST_REMOTE_CODE=True after verifying the checkpoint."
                    )
                with (Path(self.path) / ".metadata").open("rb") as metadata_file:
                    metadata = restricted_load(metadata_file)
            except FileNotFoundError as exc:
                msg = f"'{self.path}' is not a distributed checkpoint folder."
                suggested_dir = Path(self.path) / "model_and_optim"
                if (suggested_dir / ".metadata").exists():
                    msg += f" Did you mean to use '{suggested_dir}'?"
                raise FileNotFoundError(msg) from exc

            if getattr(metadata, "storage_meta", None) is None:
                metadata.storage_meta = StorageMeta()
            metadata.storage_meta.load_id = self.load_id
            self._metadata = metadata

        return self._metadata

    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        del is_coordinator
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None

    def prepare_local_plan(self, plan: dist_cp.LoadPlan) -> dist_cp.LoadPlan:
        return plan

    def prepare_global_plan(self, plans: list[dist_cp.LoadPlan]) -> list[dist_cp.LoadPlan]:
        return plans

    @property
    def checkpoint_id(self) -> str:
        return self.path

    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: str | PathLike[Any]) -> bool:
        del checkpoint_id
        return True


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module.startswith("torch") or module in ("collections", "builtins", "_collections_abc"):
            return super().find_class(module, name)
        if module.startswith("olmo_core"):
            return super().find_class("builtins", "dict") if name == "dict" else type(name, (), {})
        return super().find_class(module, name)


def restricted_load(file: Any) -> Any:
    return _RestrictedUnpickler(file).load()


def load_model(model_path: Path | str) -> dict[str, Any]:
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    if not strtobool(os.environ.get("TRUST_REMOTE_CODE", "False")):
        raise ValueError("Set TRUST_REMOTE_CODE=True after verifying the checkpoint.")

    with (Path(model_path) / ".metadata").open("rb") as metadata_file:
        metadata = restricted_load(metadata_file)
        keys = [key for key in metadata.state_dict_metadata.keys() if key.startswith("model.")]

    state_dict: dict[str, Any] = {}
    _load_state_dict(
        state_dict,
        storage_reader=RemoteFileSystemReader(model_path),
        planner=_EmptyStateDictLoadPlanner(keys=keys),
        no_dist=True,
    )
    return state_dict


def _get_tokenizer_config(config: dict[str, Any]) -> dict[str, Any]:
    if "dataset" in config:
        return config["dataset"]["tokenizer"]

    for instance_source in config.get("instance_sources", []):
        for source in instance_source.get("sources", []):
            if "tokenizer" in source:
                return source["tokenizer"]

    raise KeyError("Could not find tokenizer config under 'dataset.tokenizer' or 'instance_sources'.")


def _get_max_sequence_length(config: dict[str, Any]) -> int:
    if "train_module" in config and "max_sequence_length" in config["train_module"]:
        return config["train_module"]["max_sequence_length"]

    lengths = [source["sequence_length"] for source in config.get("instance_sources", []) if "sequence_length" in source]
    if lengths:
        return max(lengths)

    return 8192


def _required(loaded: dict[str, torch.Tensor], key: str) -> torch.Tensor:
    try:
        return loaded[key]
    except KeyError as exc:
        suffix = key.split(".", 1)[-1]
        matches = [loaded_key for loaded_key in loaded if loaded_key.endswith(suffix)]
        hint = f" Similar suffix matches: {matches[:10]}" if matches else ""
        raise KeyError(f"Missing expected OLMo Core checkpoint key '{key}'.{hint}") from exc


def _required_any(loaded: dict[str, torch.Tensor], keys: list[str]) -> torch.Tensor:
    for key in keys:
        if key in loaded:
            return loaded[key]

    layer_prefix = None
    for key in keys:
        parts = key.split(".")
        if len(parts) >= 2 and parts[0] == "blocks" and parts[1].isdigit():
            layer_prefix = f"blocks.{parts[1]}."
            break
    hint = ""
    if layer_prefix is not None:
        layer_keys = [loaded_key for loaded_key in loaded if loaded_key.startswith(layer_prefix)]
        hint = f" Layer keys: {layer_keys[:40]}"

    raise KeyError(f"Missing expected OLMo Core checkpoint key. Tried: {keys}.{hint}")


def _gdn_required(loaded: dict[str, torch.Tensor], layer_i: int, suffix: str) -> torch.Tensor:
    return _required_any(
        loaded,
        [
            f"blocks.{layer_i}.attention.{suffix}",
            f"blocks.{layer_i}.sequence_mixer.{suffix}",
            f"blocks.{layer_i}.sequence_mixer.inner.{suffix}",
            f"blocks.{layer_i}.fla.{suffix}",
            f"blocks.{layer_i}.fla.inner.{suffix}",
        ],
    )


def _gdn_weight(loaded: dict[str, torch.Tensor], layer_i: int, olmo_suffix: str, hf_suffix: str) -> torch.Tensor:
    return _required_any(
        loaded,
        [
            f"blocks.{layer_i}.attention.{olmo_suffix}",
            f"blocks.{layer_i}.sequence_mixer.{olmo_suffix}",
            f"blocks.{layer_i}.sequence_mixer.inner.{olmo_suffix}",
            f"blocks.{layer_i}.fla.{olmo_suffix}",
            f"blocks.{layer_i}.fla.inner.{olmo_suffix}",
            f"blocks.{layer_i}.attention.{hf_suffix}",
            f"blocks.{layer_i}.sequence_mixer.{hf_suffix}",
            f"blocks.{layer_i}.sequence_mixer.inner.{hf_suffix}",
            f"blocks.{layer_i}.fla.{hf_suffix}",
            f"blocks.{layer_i}.fla.inner.{hf_suffix}",
        ],
    )


def _conv_weight_for_hf(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    return tensor


def _write_custom_code(output_path: Path) -> None:
    (output_path / "configuration_pure_gdn.py").write_text(CONFIGURATION_PURE_GDN)
    (output_path / "modeling_pure_gdn.py").write_text(MODELING_PURE_GDN)


def _write_tokenizer(
    output_path: Path,
    tokenizer_id: str,
    tokenizer_config: dict[str, Any],
    max_sequence_length: int,
) -> None:
    print(f"Saving tokenizer '{tokenizer_id}' to {output_path}.")
    tokenizer = cast(Any, AutoTokenizer.from_pretrained(tokenizer_id))
    tokenizer.model_max_length = max_sequence_length
    tokenizer.pad_token_id = tokenizer_config.get("pad_token_id")
    tokenizer.bos_token_id = tokenizer_config.get("bos_token_id")
    tokenizer.eos_token_id = tokenizer_config.get("eos_token_id")
    tokenizer.save_pretrained(output_path)


def _make_config(input_config: dict[str, Any]) -> PureGDNConfig:
    model_config = input_config["model"]
    block_config = model_config["block"]
    if block_config.get("name") != "fla":
        raise ValueError(f"Expected pure GDN block name 'fla', got {block_config.get('name')!r}.")

    fla_kwargs = block_config["fla"].get("fla_layer_kwargs", {})
    sequence_mixer_config = block_config["sequence_mixer"]
    feed_forward_config = block_config["feed_forward"]
    tokenizer_config = _get_tokenizer_config(input_config)

    n_layers = model_config["n_layers"]
    n_heads = sequence_mixer_config["n_heads"]
    head_dim = fla_kwargs.get("head_dim") or (model_config["d_model"] // n_heads)
    n_value_heads = fla_kwargs.get("n_v_heads") or fla_kwargs.get("num_value_heads") or n_heads
    expand_v = fla_kwargs.get("expand_v", 2.0)
    value_head_dim = int(head_dim * expand_v)

    config = PureGDNConfig(
        vocab_size=model_config["vocab_size"],
        hidden_size=model_config["d_model"],
        intermediate_size=feed_forward_config["hidden_size"],
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=n_heads,
        hidden_act=feed_forward_config.get("activation", "silu"),
        max_position_embeddings=_get_max_sequence_length(input_config),
        initializer_range=model_config.get("init_std", 0.02),
        rms_norm_eps=block_config["layer_norm"]["eps"],
        pad_token_id=tokenizer_config.get("pad_token_id"),
        bos_token_id=tokenizer_config.get("bos_token_id"),
        eos_token_id=tokenizer_config.get("eos_token_id"),
        tie_word_embeddings=False,
        layer_types=["linear_attention"] * n_layers,
        linear_num_key_heads=n_heads,
        linear_num_value_heads=n_value_heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=value_head_dim,
        linear_conv_kernel_dim=fla_kwargs.get("conv_size", fla_kwargs.get("conv_kernel_size", 4)),
        linear_allow_neg_eigval=fla_kwargs.get("allow_neg_eigval", True),
    )
    config.architectures = ["PureGDNForCausalLM"]
    config.auto_map = {
        "AutoConfig": "configuration_pure_gdn.PureGDNConfig",
        "AutoModel": "modeling_pure_gdn.PureGDNModel",
        "AutoModelForCausalLM": "modeling_pure_gdn.PureGDNForCausalLM",
    }
    return config


def _layer_state_dict(loaded: dict[str, torch.Tensor], layer_i: int) -> dict[str, torch.Tensor]:
    prefix = f"blocks.{layer_i}"
    return {
        f"model.layers.{layer_i}.mlp.gate_proj.weight": _required(loaded, f"{prefix}.feed_forward.w1.weight"),
        f"model.layers.{layer_i}.mlp.down_proj.weight": _required(loaded, f"{prefix}.feed_forward.w2.weight"),
        f"model.layers.{layer_i}.mlp.up_proj.weight": _required(loaded, f"{prefix}.feed_forward.w3.weight"),
        f"model.layers.{layer_i}.input_layernorm.weight": _required_any(
            loaded,
            [
                f"{prefix}.fla_norm.weight",
                f"{prefix}.attention_norm.weight",
                f"{prefix}.sequence_mixer_norm.weight",
            ],
        ),
        f"model.layers.{layer_i}.post_attention_layernorm.weight": _required_any(
            loaded,
            [
                f"{prefix}.feed_forward_norm.weight",
                f"{prefix}.post_attention_norm.weight",
                f"{prefix}.ffn_norm.weight",
            ],
        ),
        f"model.layers.{layer_i}.linear_attn.A_log": _gdn_required(loaded, layer_i, "A_log"),
        f"model.layers.{layer_i}.linear_attn.dt_bias": _gdn_required(loaded, layer_i, "dt_bias"),
        f"model.layers.{layer_i}.linear_attn.q_proj.weight": _gdn_weight(
            loaded, layer_i, "w_q.weight", "q_proj.weight"
        ),
        f"model.layers.{layer_i}.linear_attn.k_proj.weight": _gdn_weight(
            loaded, layer_i, "w_k.weight", "k_proj.weight"
        ),
        f"model.layers.{layer_i}.linear_attn.v_proj.weight": _gdn_weight(
            loaded, layer_i, "w_v.weight", "v_proj.weight"
        ),
        f"model.layers.{layer_i}.linear_attn.a_proj.weight": _gdn_weight(
            loaded, layer_i, "w_a.weight", "a_proj.weight"
        ),
        f"model.layers.{layer_i}.linear_attn.b_proj.weight": _gdn_weight(
            loaded, layer_i, "w_b.weight", "b_proj.weight"
        ),
        f"model.layers.{layer_i}.linear_attn.g_proj.weight": _gdn_weight(
            loaded, layer_i, "w_g.weight", "g_proj.weight"
        ),
        f"model.layers.{layer_i}.linear_attn.o_proj.weight": _gdn_weight(
            loaded, layer_i, "w_out.weight", "o_proj.weight"
        ),
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
    n_layers = model_config["n_layers"]

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    loaded = load_model(Path(input_base_path) / "model_and_optim")["model"]
    print(f"Loaded {len(loaded)} model keys from checkpoint.")

    param_count = 0
    index_dict: dict[str, Any] = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        state_dict = _layer_state_dict(loaded, layer_i)
        for key, tensor in state_dict.items():
            index_dict["weight_map"][key] = filename
            param_count += tensor.numel()
        torch.save(state_dict, tmp_model_path / filename)
        print(f"Converted layer {layer_i} (linear_attention).")

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
    _write_custom_code(tmp_model_path)

    del state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        tokenizer_id = tokenizer_id or tokenizer_config["identifier"]
        _write_tokenizer(output_path, str(tokenizer_id), tokenizer_config, _get_max_sequence_length(input_config))

    print("Loading the checkpoint in a PureGDN model.")
    model = AutoModelForCausalLM.from_pretrained(tmp_model_path, dtype=torch.bfloat16, trust_remote_code=True)
    print("Resizing token embeddings to match tokenizer config.")
    model.resize_token_embeddings(tokenizer_config["vocab_size"])
    if hasattr(model.config, "_name_or_path"):
        del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(output_path)
    _write_custom_code(output_path)

    if tmp_cleanup:
        shutil.rmtree(tmp_model_path)

    print(f"Conversion complete. Model saved to {output_path}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert pure OLMo Core GatedDeltaNet weights to HF format.")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo Core checkpoint, containing config.json and model_and_optim/.",
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
        help="Tokenizer id/path. Defaults to the tokenizer identifier in the checkpoint config.",
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
