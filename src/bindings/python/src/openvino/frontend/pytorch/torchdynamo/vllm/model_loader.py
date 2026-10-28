# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""vLLM model loader for NNCF hook-storage checkpoints: `load_format="nncf_hooks"`.

`nncf.compress_weights()` followed by `save_pretrained()` keeps HF parameter
names but writes NNCF's own tensor layout, with each quantizer's state stored
under a hook key:

    model.layers.0.self_attn.q_proj.weight                                       u8  [out*in/2, 1]
    __nncf_hooks.post_hooks.model:layers:0:self_attn:q_proj:weight__0.0._scale   f16 [out, groups, 1]
    __nncf_hooks.post_hooks.model:layers:0:self_attn:q_proj:weight__0.0._zero_point  (asym only)

quantization.py's `"nncf"` method declares `qweight [out, in/2]`,
`scale [out, groups]` and, for asym, an unpacked `zero_point [out, groups]`.
This loader translates at load time, so the checkpoint needs no offline
conversion -- only a `quantization_config` in config.json naming the int4
modules, because vLLM fixes each layer's parameters before any weight is read.

The same tensors can instead arrive as a `.pt2`: `torch.export.export()` of
the compressed model keeps NNCF's hook modules as submodules, so its persistent
state_dict carries exactly these names. A directory with no `*.safetensors` is
read from its `*.pt2` (export with `return_dict=False`; see `_pt2_state_dict`).
config.json and the tokenizer still sit beside it, since vLLM reads the config
before any weight. Only the weights are used; vLLM builds its own model.

The form each weight takes is decided by the model vLLM built rather than
inferred: if the target layer declares a `qweight` the weight goes in packed,
otherwise it is decompressed to dense. That covers the int8 backup layers,
embeddings, lm_head, and int4 layers fused (qkv_proj, gate_up_proj) with an
int8 sibling, which quantization.py leaves unquantized.

KV-shared attention layers (Gemma-4's last `num_kv_shared_layers`) have no
`k_proj`/`v_proj`/`k_norm` in a transformers export -- HF never builds them --
but vLLM builds a full `qkv_proj` and `k_norm` on every layer. They are filled
with zeros, and only on layers vLLM itself marks `is_kv_shared_layer`: there
forward skips `k_norm`/RoPE and attention reads the target layer's cache, so
the slices are computed and discarded. Zeros rather than uninitialized memory
so that, should the OV KV-sharing substitution ever miss, PA writes zeros and
not NaN into the shared cache.

vLLM is imported only inside `register`, keeping it out of module import.
"""

import logging

logger = logging.getLogger(__name__)

LOAD_FORMAT = "nncf_hooks"
_HOOK = "__nncf_hooks.post_hooks.{0}__0.0.{1}"
_registered = None


def _hook_key(weight_name, field):
    """`model.layers.0.mlp.up_proj.weight` -> its hook-storage key."""
    return _HOOK.format(weight_name.replace(".", ":"), field)


def _unpack_uint4(packed):
    """uint4 pairs -> uint8 in [0, 15], low nibble first, as NNCF packs."""
    import torch

    return torch.stack((packed & 15, packed >> 4), dim=-1)


def _dequantize(stored, scale, zero_point, out_dtype):
    """Dense [out, in] weight from any NNCF weight mode.

    Arithmetic in the scale's dtype, cast after, as NNCF's decompressors do.
    A grouped int4 scale is [out, groups, 1]; an int8 one is [out, 1].
    """
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
    """Persistent parameters and buffers of a `torch.export.save` archive.

    The non-persistent buffers in `constants` (HF's rotary `inv_freq`) are not
    checkpoint data; `save_pretrained` never writes them either. Loading
    rebuilds the graph, so its output type must deserialize here: HF's
    `CausalLMOutputWithPast` does not, a `return_dict=False` tuple does.
    """
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
            """HF module name -> vLLM's, or None if the model drops it."""
            # Unstacked, as vLLM hands it to quant configs: q_proj stays
            # q_proj; callers resolve the fused leaf themselves.
            mapper = getattr(model, "hf_to_vllm_mapper", None)
            if mapper is None:
                return module
            mapped = mapper.get_unstacked_mapper().apply_list([module])
            return mapped[0] if mapped else None

        def _target_is_quantized(self, model, params, module):
            """Does the vLLM layer this HF module loads into hold a qweight?"""
            module = self._to_vllm(model, module)
            if module is None:
                # The model drops this weight; it has no layer to pack into.
                return False
            parent, _, leaf = module.rpartition(".")
            mapping = getattr(model, "packed_modules_mapping", {})
            for fused, sources in mapping.items():
                if leaf in sources:
                    leaf = fused
                    break
            return f"{parent}.{leaf}.qweight" in params

        def _kv_shared_zeros(self, model, params, seen, out_dtype):
            """Zeros for the k/v/k_norm a KV-shared layer has no weights for.

            Keyed off each attention block whose q_proj arrived; shapes come
            from the layer vLLM built.
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
                    # Not ours to fill (e.g. MLA has q_proj and no k_proj); a
                    # real gap still fails load_weights' missing check.
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
        def _shards(model_dir):
            """`(keys, get_tensor)` per checkpoint file: safetensors, else pt2."""
            paths = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
            if paths:
                for path in paths:
                    with safe_open(path, framework="pt") as f:
                        yield set(f.keys()), f.get_tensor
                return
            for path in sorted(glob.glob(os.path.join(model_dir, "*.pt2"))):
                state = _pt2_state_dict(path)
                yield set(state), state.__getitem__

        def _weights(self, model, model_config):
            params = dict(model.named_parameters())
            out_dtype = model_config.dtype
            seen = set()
            with contextlib.closing(self._shards(model_config.model)) as shards:
                for keys, get_tensor in shards:
                    seen.update(keys)
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
                            # The packed buffer is already row-major over
                            # (out, in/2), so the reshapes are free.
                            out_features, groups = scale.shape[:2]
                            yield f"{module}.qweight", stored.reshape(out_features, -1)
                            yield f"{module}.scale", scale.reshape(out_features, groups)
                            if zp is not None:
                                # quantization.py stores the zero point
                                # unpacked so it concatenates like `scale`.
                                yield (f"{module}.zero_point",
                                       _unpack_uint4(zp).reshape(out_features, groups))
                        else:
                            yield name, _dequantize(stored, scale, zp, out_dtype)
            yield from self._kv_shared_zeros(model, params, seen, out_dtype)

        def load_weights(self, model, model_config):
            loaded = model.load_weights(self._weights(model, model_config))
            # DefaultModelLoader skips this check for quantized models; here a
            # missed key means a layout this loader does not understand.
            missing = {n for n, _ in model.named_parameters()} - set(loaded or ())
            if missing:
                raise ValueError(
                    f"[OV plugin] {LOAD_FORMAT}: not initialized from "
                    f"checkpoint: {sorted(missing)}")

    return NNCFHooksModelLoader


def register():
    """Add `load_format="nncf_hooks"` to vLLM. Inert unless a run asks for it.

    Memoized: vLLM warns when a load format is registered twice.
    """
    global _registered
    if _registered is None:
        try:
            _registered = _define_loader()
        except Exception as _e:
            logger.debug("[OV plugin] model loader not registered: %s", _e)
            return None
        logger.debug("[OV plugin] registered load format '%s'", LOAD_FORMAT)
    return _registered
