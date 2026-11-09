# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM model loader for NNCF hook-storage checkpoints (`load_format="nncf_hooks"`), safetensors or `.pt2`.

Translates NNCF's layout to quantization.py's "nncf" parameters at load time; config.json still needs a
`quantization_config`, since vLLM fixes each layer's parameters before reading any weight.
"""

import logging

from .quantization import _unpack_uint4

logger = logging.getLogger(__name__)

LOAD_FORMAT = "nncf_hooks"
_HOOK = "__nncf_hooks.post_hooks.{0}__0.0.{1}"
_registered = None


def _hook_key(weight_name, field):
    """`model.layers.0.mlp.up_proj.weight` -> its hook-storage key."""
    return _HOOK.format(weight_name.replace(".", ":"), field)


def _dequantize(stored, scale, zero_point, out_dtype):
    """Dense [out, in] weight from any NNCF mode, computed in the scale's dtype as NNCF's decompressors do."""
    out_features = scale.shape[0]
    if scale.dim() == 3:
        groups = scale.shape[1]
        weight = _unpack_uint4(stored).reshape(out_features, groups, -1)
        weight = weight.type(scale.dtype)
        if zero_point is None:
            # Symmetric nibbles carry a +8 bias to stay unsigned.
            weight = weight - 8
        else:
            zp = _unpack_uint4(zero_point).reshape(out_features, groups, 1)
            weight = weight - zp.type(scale.dtype)
    else:
        weight = stored.type(scale.dtype)
        if zero_point is not None:
            weight = weight - zero_point.type(scale.dtype)
    weight = weight * scale
    return weight.reshape(out_features, -1).type(out_dtype)


def _pt2_state_dict(path):
    """Persistent parameters and buffers of a `torch.export.save` archive (exported with return_dict=False)."""
    import torch

    try:
        exported = torch.export.load(path)
    except Exception as exc:
        raise ValueError(
            f"[OV plugin] {LOAD_FORMAT}: cannot read {path}: {exc}. Export "
            "with kwargs return_dict=False so the output is a plain tuple") from exc
    return exported.state_dict


def _define_loader():
    import contextlib
    import glob
    import os

    from safetensors import safe_open
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader(LOAD_FORMAT)
    class NNCFHooksModelLoader(DefaultModelLoader):
        """DefaultModelLoader with NNCF's hook layout translated in transit."""

        @staticmethod
        def _to_vllm(model, module):
            """HF module name -> vLLM's unstacked one (q_proj stays q_proj), or None if the model drops it."""
            mapper = getattr(model, "hf_to_vllm_mapper", None)
            if mapper is None:
                return module
            mapped = mapper.get_unstacked_mapper().apply_list([module])
            return mapped[0] if mapped else None

        def _target_is_quantized(self, model, params, module):
            """Does the vLLM layer this HF module loads into hold a qweight?"""
            module = self._to_vllm(model, module)
            if module is None:
                return False
            parent, _, leaf = module.rpartition(".")
            mapping = getattr(model, "packed_modules_mapping", {})
            for fused, sources in mapping.items():
                if leaf in sources:
                    leaf = fused
                    break
            return f"{parent}.{leaf}.qweight" in params

        def _kv_shared_zeros(self, model, params, seen, out_dtype):
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
        def _checkpoint(model_dir):
            """`(keys, get_tensor)` over all safetensors (else pt2) files; a scale may sit in another shard."""
            paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
            if paths:
                with contextlib.ExitStack() as stack:
                    owner = {}
                    for path in paths:
                        shard = stack.enter_context(safe_open(path, framework="pt"))
                        owner.update(dict.fromkeys(shard.keys(), shard))
                    yield set(owner), lambda key: owner[key].get_tensor(key)
                return
            state = {}
            for path in sorted(glob.glob(os.path.join(model_dir, "*.pt2"))):
                state.update(_pt2_state_dict(path))
            yield set(state), state.__getitem__

        def _weights(self, model, model_config):
            params = dict(model.named_parameters())
            out_dtype = model_config.dtype
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

                    if scale.dim() == 3 and self._target_is_quantized(
                            model, params, module):
                        # Already row-major over (out, in/2), so the reshapes are free.
                        out_features, groups = scale.shape[:2]
                        yield f"{module}.qweight", stored.reshape(out_features, -1)
                        yield f"{module}.scale", scale.reshape(out_features, groups)
                        if zp is not None:
                            # Unpacked, so it concatenates like `scale`.
                            yield (f"{module}.zero_point",
                                   _unpack_uint4(zp).reshape(out_features, groups))
                    else:
                        yield name, _dequantize(stored, scale, zp, out_dtype)
                yield from self._kv_shared_zeros(model, params, keys, out_dtype)

        def load_weights(self, model, model_config):
            loaded = model.load_weights(self._weights(model, model_config))
            # DefaultModelLoader skips this check for quantized models; here it catches unknown layouts.
            missing = {n for n, _ in model.named_parameters()} - set(loaded or ())
            if missing:
                raise ValueError(
                    f"[OV plugin] {LOAD_FORMAT}: not initialized from "
                    f"checkpoint: {sorted(missing)}")

    return NNCFHooksModelLoader


def register():
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
