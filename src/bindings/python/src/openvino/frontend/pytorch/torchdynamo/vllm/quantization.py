# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""Weight-only integer quantization for vLLM, decoded in plain torch.

Registers a `"quant_method": "nncf"` config so an NNCF checkpoint loads through
vLLM's own weight loader instead of being patched over a full-precision model.
Scope: weight-only, sym or asym, per-group or per-channel, 4-bit (two nibbles
per uint8, low nibble first) or 8-bit. Partial quantization is the normal case
-- `quantized_modules` names what NNCF compressed and everything else stays
dense. No execution dtype is named anywhere: the multiply happens in the
scale's dtype and the result casts to the layer's `params_dtype`.

Lives in the OV plugin because `vllm.general_plugins` loads in every process.
The classes are built lazily by `_define_classes`, which keeps torch and vLLM
out of module import; `_build_classes` memoizes them and the module-level
`__getattr__` resolves their names, which together is what lets an NNCFConfig
pickle into a spawned engine process.
"""

import logging
import threading

logger = logging.getLogger(__name__)

# Lazily built by `_build_classes`, in the order `_define_classes` returns them.
_LAZY_CLASSES = ("NNCFConfig", "NNCFLinearMethod")
_classes = None
# Plain Lock: `_define_classes` reaches NNCFLinearMethod through a closure cell,
# never as a module attribute, so it cannot re-enter this.
_classes_lock = threading.Lock()

# Storage dtype per `bits`; 4-bit arrives packed two-per-byte, 8-bit one.
_STORAGE = {4: "uint8", 8: "int8"}


def _unpack_int4(packed):
    """Symmetric int4 -> int8 in [-8, 7], gaining a trailing dim of 2.

    Low nibble of byte k first, as NNCF packs; the -8 undoes the unsigned bias.
    """
    import torch

    low = torch.bitwise_and(packed, 15)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack((low, high), dim=-1).type(torch.int8) - 8


def _unpack_uint4(packed):
    """Asymmetric int4 -> uint8 in [0, 15], gaining a trailing dim of 2.

    Same nibble order as `_unpack_int4`; the offset lives in a zero point.
    """
    import torch

    low = torch.bitwise_and(packed, 15)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack((low, high), dim=-1)


def _dequantize(qweight, scale, bits, group_size, out_dtype, zero_point=None):
    """Decompress to a dense [out, in] weight in `out_dtype`.

    Multiply in the scale's dtype and cast after, and subtract `zero_point`
    before the multiply: that order is what matches NNCF bit for bit.
    """
    out_features = qweight.shape[0]
    if bits == 4:
        in_features = qweight.shape[1] * 2
        weight = (_unpack_int4(qweight) if zero_point is None
                  else _unpack_uint4(qweight))
    else:
        in_features = qweight.shape[1]
        weight = qweight

    if group_size > 0:
        groups = scale.shape[1]
        weight = weight.reshape(out_features, groups, in_features // groups)
        weight = weight.type(scale.dtype)
        if zero_point is not None:
            weight = weight - zero_point.type(scale.dtype).unsqueeze(-1)
        weight = weight * scale.unsqueeze(-1)
        weight = weight.reshape(out_features, in_features)
    else:
        # Per output channel: scale is [out, 1] and broadcasts as stored.
        weight = weight.reshape(out_features, in_features)
        weight = weight.type(scale.dtype)
        if zero_point is not None:
            weight = weight - zero_point.type(scale.dtype)
        weight = weight * scale
    return weight.type(out_dtype)


def _define_classes():
    """Define the config/method pair against an importable vLLM.

    Call `_build_classes` instead: two builds would register "nncf" twice, to
    two different class objects.
    """
    import torch
    from torch.nn import functional
    from vllm.model_executor.layers.linear import (
        LinearBase,
        LinearMethodBase,
        UnquantizedLinearMethod,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config
    from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
    from vllm.model_executor.parameter import (
        ChannelQuantScaleParameter,
        GroupQuantScaleParameter,
        ModelWeightParameter,
        PackedvLLMParameter,
    )

    @register_quantization_config("nncf")
    class NNCFConfig(QuantizationConfig):
        """Weight-only integer compression, as written by NNCF."""

        def __init__(self, bits, group_size, sym, scale_dtype, quantized_modules):
            super().__init__()
            if bits not in _STORAGE:
                raise ValueError(f"bits={bits} is not one of {sorted(_STORAGE)}")
            self.bits = bits
            self.sym = sym
            self.group_size = group_size
            self.scale_dtype = getattr(torch, scale_dtype)
            # Empty means "every linear layer".
            self.quantized_modules = set(quantized_modules or ())

        def __repr__(self):
            return (f"NNCFConfig(bits={self.bits}, "
                    f"{'sym' if self.sym else 'asym'}, "
                    f"group_size={self.group_size}, "
                    f"scale_dtype={self.scale_dtype}, "
                    f"modules={len(self.quantized_modules) or 'all'})")

        @classmethod
        def get_name(cls):
            return "nncf"

        @classmethod
        def get_supported_act_dtypes(cls):
            # Weight-only: every float the model may run in is supported.
            return [torch.float32, torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls):
            # No kernel requirement of its own; the decode is plain torch.
            return 60

        @staticmethod
        def get_config_filenames():
            # Self-describing checkpoint: everything is in config.json.
            return []

        @classmethod
        def from_config(cls, config):
            return cls(
                bits=config.get("bits", 4),
                group_size=config.get("group_size", -1),
                sym=config.get("sym", True),
                scale_dtype=config.get("scale_dtype", "float16"),
                quantized_modules=config.get("quantized_modules", ()),
            )

        def apply_vllm_mapper(self, hf_to_vllm_mapper):
            # `quantized_modules` holds checkpoint names, but `_is_quantized`
            # is asked with vLLM prefixes; models that rename (gemma4_mm.py:
            # model.language_model. -> language_model.model.) would otherwise
            # match nothing and silently build every layer dense.
            self.quantized_modules = set(
                hf_to_vllm_mapper.apply_list(sorted(self.quantized_modules)))

        def _is_quantized(self, prefix):
            """Is the vLLM layer at `prefix` one the checkpoint compressed?

            A fused layer (qkv_proj <- q/k/v_proj) counts only when all of its
            packed_modules_mapping sources do, since it decodes as one tensor.
            """
            if not self.quantized_modules:
                return True
            parent, _, leaf = prefix.rpartition(".")
            sources = self.packed_modules_mapping.get(leaf)
            names = ([f"{parent}.{s}" for s in sources] if sources else [prefix])
            return all(n in self.quantized_modules for n in names)

        def get_quant_method(self, layer, prefix):
            # None would trip linear.py's "All linear layers should support
            # quant method"; a dense layer has to get Unquantized instead.
            if isinstance(layer, LinearBase):
                if self._is_quantized(prefix):
                    return NNCFLinearMethod(self)
                return UnquantizedLinearMethod()
            return None

    class NNCFLinearMethod(LinearMethodBase):
        """Declares the compressed parameters, then decodes them per forward."""

        def __init__(self, quant_config):
            self.quant_config = quant_config

        def create_weights(self, layer, input_size_per_partition,
                           output_partition_sizes, input_size, output_size,
                           params_dtype, **extra_weight_attrs):
            config = self.quant_config
            weight_loader = extra_weight_attrs.get("weight_loader")
            output_size_per_partition = sum(output_partition_sizes)
            packed_factor = 8 // config.bits

            if divmod(input_size_per_partition, packed_factor)[1]:
                raise ValueError(
                    f"input_size_per_partition {input_size_per_partition} does "
                    f"not divide into {config.bits}-bit groups of "
                    f"{packed_factor}")
            storage = torch.uint8 if not config.sym else getattr(torch, _STORAGE[config.bits])
            data = torch.empty(output_size_per_partition,
                               input_size_per_partition // packed_factor,
                               dtype=storage)

            if config.bits == 4:
                # packed_dim=1: nibble pairs run along the input dim, so a
                # row-parallel shard has to halve its offsets.
                qweight = PackedvLLMParameter(
                    data=data, input_dim=1, output_dim=0, packed_dim=1,
                    packed_factor=packed_factor, weight_loader=weight_loader)
            else:
                qweight = ModelWeightParameter(
                    data=data, input_dim=1, output_dim=0,
                    weight_loader=weight_loader)

            # The zero point has the scale's shape, so it shards and
            # concatenates down the same path; only the dtype differs.
            def _per_group(dtype):
                if config.group_size > 0:
                    if divmod(input_size_per_partition, config.group_size)[1]:
                        raise ValueError(
                            f"input_size_per_partition "
                            f"{input_size_per_partition} is not a multiple of "
                            f"group_size {config.group_size}")
                    return GroupQuantScaleParameter(
                        data=torch.empty(
                            output_size_per_partition,
                            input_size_per_partition // config.group_size,
                            dtype=dtype),
                        input_dim=1, output_dim=0, weight_loader=weight_loader)
                return ChannelQuantScaleParameter(
                    data=torch.empty(output_size_per_partition, 1, dtype=dtype),
                    output_dim=0, weight_loader=weight_loader)

            layer.register_parameter("qweight", qweight)
            layer.register_parameter("scale", _per_group(config.scale_dtype))
            if not config.sym:
                # Unpacked, one byte per group: packing would save ~3% and
                # break output-dim concatenation of fused layers.
                layer.register_parameter("zero_point", _per_group(torch.uint8))
            # The layer is the only thing that knows what --dtype resolved to.
            layer.nncf_out_dtype = params_dtype

        def process_weights_after_loading(self, layer):
            # Deliberately empty: the unquantized CPU path would swap in a
            # oneDNN GEMM over `weight`, which this method never creates.
            return

        def apply(self, layer, inputs, bias=None):
            config = self.quant_config
            weight = _dequantize(
                layer.qweight, layer.scale, config.bits, config.group_size,
                layer.nncf_out_dtype,
                zero_point=None if config.sym else layer.zero_point)
            return functional.linear(inputs, weight, bias)

    # Pickle stores a class as module + qualname, so the qualname must not read
    # `_define_classes.<locals>.…`; the module `__getattr__` resolves the rest.
    for cls in (NNCFConfig, NNCFLinearMethod):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__

    return NNCFConfig, NNCFLinearMethod


def _build_classes():
    """`_define_classes`, memoized. Returns `(NNCFConfig, NNCFLinearMethod)`.

    Required rather than an optimization: pickle's `save_global` re-reads the
    reference it writes and refuses unless it yields the same object.
    """
    global _classes
    if _classes is not None:
        return _classes
    with _classes_lock:
        # Re-checked: two threads can both pass the test above.
        if _classes is None:
            _classes = _define_classes()
    return _classes


def __getattr__(name):
    """Resolve the lazily built classes as module attributes (PEP 562).

    Pickle finds a class by module plus qualname, so these names have to
    resolve here; building on demand keeps torch and vLLM out of import.
    """
    if name not in _LAZY_CLASSES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return _build_classes()[_LAZY_CLASSES.index(name)]
    except Exception as exc:
        # AttributeError keeps `hasattr`/`getattr(..., default)` working, which
        # an escaping ImportError would break; `from exc` keeps the cause.
        raise AttributeError(
            f"module {__name__!r} cannot provide {name!r}: {exc}") from exc


def register():
    """Add the method to vLLM's registry. Inert unless a checkpoint asks for it.

    Safe to call repeatedly and in any process: the build behind it is memoized.
    """
    try:
        config_cls, _method_cls = _build_classes()
    except Exception as _e:
        logger.debug("[OV plugin] quantization method not registered: %s", _e)
        return None
    logger.debug("[OV plugin] registered quantization method '%s'",
                 config_cls.get_name())
    return config_cls
