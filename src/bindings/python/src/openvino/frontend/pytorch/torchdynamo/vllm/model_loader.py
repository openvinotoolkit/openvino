# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""vLLM model loader for NNCF hook-storage checkpoints (`load_format="nncf_hooks"`), safetensors or `.pt2`.

Translates NNCF's layout to quantization.py's "nncf" parameters at load time; config.json still needs a
`quantization_config`, since vLLM fixes each layer's parameters before reading any weight.
"""

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from .quantization import _unpack_uint4

if TYPE_CHECKING:
    import torch
    from vllm.config import ModelConfig

logger = logging.getLogger(__name__)

LOAD_FORMAT = "nncf_hooks"
_HOOK = "__nncf_hooks.post_hooks.{0}__0.0.{1}"
_registered: type[Any] | None = None


def _hook_key(weight_name: str, field: str) -> str:
    """`model.layers.0.mlp.up_proj.weight` -> its hook-storage key."""
    return _HOOK.format(weight_name.replace(".", ":"), field)


def _dequantize(stored: "torch.Tensor", scale: "torch.Tensor", zero_point: "torch.Tensor | None",
                out_dtype: "torch.dtype") -> "torch.Tensor":
    """Dense [..., out, in] weight from any NNCF mode, computed in the scale's dtype as NNCF's decompressors do.

    Grouped int4 scales are [..., out, groups, 1]; leading dims are fused MoE experts (3D `experts.gate_up_proj`).
    """
    if scale.dim() >= 3:
        rows = scale.shape[:-2]
        weight = _unpack_uint4(stored).reshape(*scale.shape[:-1], -1)
        weight = weight.type(scale.dtype)
        if zero_point is None:
            # Symmetric nibbles carry a +8 bias to stay unsigned.
            weight = weight - 8
        else:
            zp = _unpack_uint4(zero_point).reshape(scale.shape)
            weight = weight - zp.type(scale.dtype)
    else:
        rows = scale.shape[:1]
        weight = stored.type(scale.dtype)
        if zero_point is not None:
            weight = weight - zero_point.type(scale.dtype)
    weight = weight * scale
    return weight.reshape(*rows, -1).type(out_dtype)


def _pt2_state_dict(path: str) -> "dict[str, torch.Tensor]":
    """Persistent parameters and buffers of a `torch.export.save` archive, read without rebuilding its graph.

    `torch.export.load` would deserialize the program too, which fails on output types pytree cannot rebuild
    (HF's `MoeCausalLMOutputWithPast` when Qwen2-MoE cannot be exported with return_dict=False); only the
    weights are used here.
    """
    from torch.export.pt2_archive._package import PT2ArchiveReader, _load_state_dict

    state: "dict[str, torch.Tensor]" = {}
    try:
        with PT2ArchiveReader(path) as archive:
            for file in archive.get_file_names():
                if file.startswith("models/") and file.endswith(".json"):
                    weights = _load_state_dict(archive, file.removeprefix("models/").removesuffix(".json"))
                    if isinstance(weights, bytes):
                        raise ValueError("legacy pickled weights; re-save with torch.export.save")
                    state.update(weights)
    except Exception as exc:
        raise ValueError(f"[OV plugin] {LOAD_FORMAT}: cannot read weights from {path}: {exc}") from exc
    return state


def _define_loader() -> type[Any]:
    import contextlib
    import glob
    import os

    from safetensors import safe_open
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
    from vllm.model_executor.model_loader.weight_utils import maybe_remap_moe_expert_param_name

    @register_model_loader(LOAD_FORMAT)
    class NNCFHooksModelLoader(DefaultModelLoader):
        """DefaultModelLoader with NNCF's hook layout translated in transit."""

        @staticmethod
        def _to_vllm(model: "torch.nn.Module", module: str) -> str | None:
            """HF module name -> vLLM's unstacked one (q_proj stays q_proj), or None if the model drops it."""
            mapper = getattr(model, "hf_to_vllm_mapper", None)
            if mapper is None:
                return module
            mapped = mapper.get_unstacked_mapper().apply_list([module])
            return mapped[0] if mapped else None

        def _target_is_quantized(self, model: "torch.nn.Module", params: "dict[str, torch.nn.Parameter]",
                                 module: str) -> bool:
            """Does the vLLM layer this HF module loads into hold a qweight?"""
            vllm_module = self._to_vllm(model, module)
            if vllm_module is None:
                return False
            parent, _, leaf = vllm_module.rpartition(".")
            mapping = getattr(model, "packed_modules_mapping", {})
            for fused, sources in mapping.items():
                if leaf in sources:
                    leaf = fused
                    break
            return f"{parent}.{leaf}.qweight" in params

        def _experts_are_quantized(self, model: "torch.nn.Module", params: "dict[str, torch.nn.Parameter]",
                                   module: str) -> bool:
            """Was the FusedMoE that `...experts.gate_up_proj` / `...experts.down_proj` loads into built int4?"""
            experts = self._to_vllm(model, module.rpartition(".")[0])
            return experts is not None and any(
                f"{experts}{routed}.w13_qweight" in params for routed in ("", ".routed_experts"))

        @staticmethod
        def _expert_shards(module: str, stored: "torch.Tensor", scale: "torch.Tensor",
                           zero_point: "torch.Tensor | None") -> "Iterator[tuple[str, torch.Tensor]]":
            """Per-expert 2D `experts.{e}.{gate,up,down}_proj.{qweight,scale,zero_point}` from NNCF's fused 3D layout.

            2D, so FusedMoE's per-expert mapping takes them; a 3D tensor would hit its fused-checkpoint path,
            which reads the packed in/2 dim as a transposed hidden dim. Scales are [experts, out, groups, 1].
            """
            if zero_point is None:
                raise ValueError(f"[OV plugin] {LOAD_FORMAT}: {module} is symmetric; MoE experts need int4 asym")
            experts, out_features, groups = scale.shape[:3]
            fields = {
                "qweight": stored.reshape(experts, out_features, -1),
                "scale": scale.reshape(experts, out_features, groups),
                "zero_point": _unpack_uint4(zero_point).reshape(experts, out_features, groups),
            }
            parent, _, leaf = module.rpartition(".")
            if leaf == "gate_up_proj":
                # HF's fused gate_up is [gate; up] along out, as vLLM's w13 is [w1; w3].
                half = out_features // 2
                projections = (("gate_proj", slice(0, half)), ("up_proj", slice(half, None)))
            else:
                projections = ((leaf, slice(None)),)
            for expert in range(experts):
                for projection, rows in projections:
                    for field, tensor in fields.items():
                        yield f"{parent}.{expert}.{projection}.{field}", tensor[expert, rows]

        def _kv_shared_zeros(self, model: "torch.nn.Module", params: "dict[str, torch.nn.Parameter]",
                             seen: set[str], out_dtype: "torch.dtype") -> "Iterator[tuple[str, torch.Tensor]]":
            """Zeros for the k/v/k_norm vLLM builds on a KV-shared layer (HF has none); computed, then discarded.

            Zeros rather than empty memory, so a missed KV-sharing substitution writes zeros, not NaN, to the cache.
            """
            import torch

            modules = dict(model.named_modules())
            for name in sorted(seen):
                if not name.endswith(".q_proj.weight"):
                    continue
                base = name.removesuffix(".q_proj.weight")
                wanted = [f"{base}.{p}.weight" for p in ("k_proj", "v_proj", "k_norm")]
                missing = [w for w in wanted if w not in seen]
                if not missing:
                    continue
                vllm_base = self._to_vllm(model, f"{base}.q_proj")
                attn = modules.get(vllm_base.rpartition(".")[0]) if vllm_base else None
                if not getattr(attn, "is_kv_shared_layer", False):
                    # Not ours to fill (e.g. MLA); a real gap still fails the missing-weights check.
                    continue
                assert attn is not None
                qkv = attn.qkv_proj
                if self._target_is_quantized(model, params, f"{base}.k_proj"):
                    raise ValueError(
                        f"[OV plugin] {LOAD_FORMAT}: {base}.qkv_proj was built "
                        "int4 but its k/v are absent; drop the KV-shared "
                        "k_proj/v_proj from quantized_modules")
                # Unsharded: the qkv weight_loader takes this rank's slice.
                kv_shape = (qkv.total_num_kv_heads * qkv.head_size, qkv.input_size)
                for weight_name in missing:
                    if ".k_norm." in weight_name:
                        yield weight_name, torch.zeros_like(attn.k_norm.weight)
                    else:
                        yield weight_name, torch.zeros(kv_shape, dtype=out_dtype)

        @staticmethod
        @contextlib.contextmanager
        def _checkpoint(model_dir: str) -> "Iterator[tuple[set[str], Callable[[str], torch.Tensor]]]":
            """`(keys, get_tensor)` over all safetensors (else pt2) files; a scale may sit in another shard."""
            paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
            if paths:
                with contextlib.ExitStack() as stack:
                    owner: dict[str, Any] = {}
                    for path in paths:
                        shard = stack.enter_context(safe_open(path, framework="pt"))
                        owner.update(dict.fromkeys(shard.keys(), shard))
                    yield set(owner), lambda key: owner[key].get_tensor(key)
                return
            state: "dict[str, torch.Tensor]" = {}
            for path in sorted(glob.glob(os.path.join(model_dir, "*.pt2"))):
                state.update(_pt2_state_dict(path))
            yield set(state), state.__getitem__

        def _weights(self, model: "torch.nn.Module",
                     model_config: "ModelConfig") -> "Iterator[tuple[str, torch.Tensor]]":
            params = dict(model.named_parameters())
            out_dtype = model_config.dtype
            quant_config = getattr(model_config.hf_config, "quantization_config", None) or {}
            bits = quant_config.get("bits", 4)
            with self._checkpoint(model_config.model) as (keys, get_tensor):
                for name in sorted(keys):
                    if name.startswith("__nncf_hooks."):
                        # Consumed alongside the weight it belongs to.
                        continue
                    scale_key = _hook_key(name, "_scale")
                    if scale_key not in keys:
                        # Norms and anything else NNCF left alone.
                        yield name, get_tensor(name)
                        continue

                    stored = get_tensor(name)
                    scale = get_tensor(scale_key)
                    zp_key = _hook_key(name, "_zero_point")
                    zp = get_tensor(zp_key) if zp_key in keys else None
                    module = name.removesuffix(".weight")

                    if scale.dim() == 4 and self._experts_are_quantized(model, params, module):
                        yield from self._expert_shards(module, stored, scale, zp)
                    elif self._target_is_quantized(model, params, module):
                        # Already row-major over (out, in) or (out, in/2), so the reshapes are free.
                        out_features = scale.shape[0]
                        yield f"{module}.qweight", stored.reshape(out_features, -1)
                        yield f"{module}.scale", scale.reshape(out_features, -1)
                        if zp is not None:
                            # int4 zero points are packed; unpacked, they concatenate like `scale`.
                            zp = _unpack_uint4(zp) if bits == 4 else zp
                            yield f"{module}.zero_point", zp.reshape(out_features, -1)
                    else:
                        yield name, _dequantize(stored, scale, zp, out_dtype)
                yield from self._kv_shared_zeros(model, params, keys, out_dtype)

        def load_weights(self, model: "torch.nn.Module", model_config: "ModelConfig") -> None:
            loaded = model.load_weights(self._weights(model, model_config))
            # DefaultModelLoader skips this check for quantized models; here it catches unknown layouts.
            params = dict(model.named_parameters())
            # MoERunner reports `experts.w13_weight`; the parameter is `experts.routed_experts.w13_weight`.
            # vLLM's remap knows a fixed suffix list, which lacks NNCF's `w13_zero_point`.
            loaded = {maybe_remap_moe_expert_param_name(n, params) for n in loaded or ()}
            loaded |= {routed for n in loaded if n not in params
                       and (routed := n.replace(".experts.", ".experts.routed_experts.", 1)) in params}
            missing = set(params) - loaded
            if missing:
                raise ValueError(
                    f"[OV plugin] {LOAD_FORMAT}: not initialized from "
                    f"checkpoint: {sorted(missing)}")

    return NNCFHooksModelLoader


def register() -> type[Any] | None:
    """Add `load_format="nncf_hooks"` to vLLM, once (vLLM warns on a second registration)."""
    global _registered
    if _registered is None:
        try:
            _registered = _define_loader()
        except Exception as _e:
            logger.debug("[OV plugin] model loader not registered: %s", _e)
            return None
        logger.debug("[OV plugin] registered load format '%s'", LOAD_FORMAT)
    return _registered
