# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

"""Weight-only integer quantization for vLLM, decoded in plain torch.

Registers a vLLM quantization method so a checkpoint whose `config.json` says

    "quantization_config": {"quant_method": "nncf", "bits": 4, "sym": true,
                            "group_size": 64, "scale_dtype": "float16"}

loads through vLLM's own weight loader. Without it such a checkpoint is
rejected at startup ("Unknown quantization method"), and a runner has to load
the full-precision model and patch compressed weights over the top -- which
costs the full-precision allocation, bypasses tensor-parallel sharding, and
re-implements the fused-layer concatenation vLLM already does.

Why this lives in the OV plugin rather than in a runner: `vllm.general_plugins`
is loaded in *every* process (parent, engine core, workers), which is what the
registry needs -- a method registered in the parent only is invisible to the
worker that actually builds the model.

Scope: weight-only, symmetric or asymmetric, per-group or per-channel, 4-bit
(two nibbles per uint8, low nibble first) or 8-bit. An asymmetric checkpoint
additionally carries a `zero_point` beside every `scale`, and declares itself
with `"sym": false`.

Symmetric and asymmetric differ only in where the offset lives. Symmetric
packing biases the stored nibble by +8 to keep it unsigned and the decoder
subtracts that constant back out; asymmetric stores the raw nibble and a
per-group zero point, and the decoder subtracts *that*. Hence one `bits`
parameter and two unpackers rather than two decode paths.

Dtype independence
------------------
Nothing here names an execution dtype. The scale keeps the dtype the checkpoint
stored it in -- the multiply happens there, as NNCF's own decompressors do --
and the result is cast to the layer's `params_dtype`, whatever vLLM resolved
`--dtype` to. One checkpoint therefore runs fp32, fp16 or bf16 unchanged.

The decode is deliberately plain torch ops rather than a custom op: dynamo
traces them into the FX graph, and OpenVINO's PyTorch frontend sees
`Constant(u4/i4/u8/i8) -> Convert -> Multiply`, which
MarkCompressedWeightsCast marks as a decompression subgraph so the weights
reach the CPU plugin still compressed.
"""

import logging

logger = logging.getLogger(__name__)

# What `bits` this decoder understands, and the torch dtype each is stored as.
# 4-bit arrives packed two-per-byte, so its storage dtype is uint8 and the
# parameter carries a packing factor; 8-bit is stored one-per-byte.
_STORAGE = {4: "uint8", 8: "int8"}


def _unpack_int4(packed):
    """Unpack symmetric int4; returns int8 with values in [-8, 7].

    Element 2k is the low nibble of byte k and element 2k+1 the high nibble,
    which is the order NNCF packs in. The -8 undoes the bias the symmetric
    packing adds to keep the stored nibble unsigned. Gains a trailing dim of
    2; the caller reshapes.
    """
    import torch

    low = torch.bitwise_and(packed, 15)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack((low, high), dim=-1).type(torch.int8) - 8


def _unpack_uint4(packed):
    """Unpack asymmetric int4; returns uint8 with values in [0, 15].

    The same nibble order as `_unpack_int4`, without the -8: the asymmetric
    scheme carries the offset in a zero point instead of in the stored value.
    Gains a trailing dim of 2; the caller reshapes.
    """
    import torch

    low = torch.bitwise_and(packed, 15)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack((low, high), dim=-1)


def _dequantize(qweight, scale, bits, group_size, out_dtype, zero_point=None):
    """Decompress to a dense [out, in] weight in `out_dtype`.

    The multiply happens in the scale's dtype and the cast comes after, which
    is what keeps the result bit-identical to NNCF's decompressor rather than
    merely close to it.

    `zero_point` is the asymmetric offset, one per group (or per output channel
    when `group_size <= 0`), stored unpacked. The cast to the scale's dtype
    happens *before* the subtraction, matching NNCF's `decompress_asymmetric`.
    That order is load-bearing twice over: subtracting in uint8 would wrap
    around, and doing the multiply in anything but the scale's dtype would
    drift off NNCF's result.
    """
    out_features = qweight.shape[0]
    if bits == 4:
        in_features = qweight.shape[1] * 2
        weight = _unpack_int4(qweight) if zero_point is None \
            else _unpack_uint4(qweight)
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


def _build_classes():
    """Define the config/method pair against an importable vLLM.

    Done inside a function so importing this module costs nothing until the
    plugin actually registers, and so a vLLM too old for these symbols fails
    at registration with a logged warning rather than at import.
    """
    import torch
    import torch.nn.functional as F
    from vllm.model_executor.layers.linear import LinearBase, LinearMethodBase
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
            # Empty means "every linear layer", which is what a checkpoint
            # without an explicit list implies.
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
            # Activations are never quantized here, so every float the model
            # may run in is supported; the dequantized weight follows.
            return [torch.float32, torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls):
            # Only consulted where a device capability exists; the decode is
            # plain torch and has no kernel requirement of its own.
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

        def _is_quantized(self, prefix):
            """Is the layer at `prefix` one the checkpoint compressed?

            `prefix` names the *vLLM* layer, which may be a fusion of several
            checkpoint modules (qkv_proj <- q_proj/k_proj/v_proj). vLLM hands
            the model's packed_modules_mapping to every quant config for
            exactly this translation. A fused layer counts as quantized only
            when all of its sources are, since it is decoded as one tensor.
            """
            if not self.quantized_modules:
                return True
            parent, _, leaf = prefix.rpartition(".")
            sources = self.packed_modules_mapping.get(leaf)
            names = ([f"{parent}.{s}" for s in sources] if sources else [prefix])
            return all(n in self.quantized_modules for n in names)

        def get_quant_method(self, layer, prefix):
            # Embeddings and the lm_head fall through to None, i.e. vLLM's
            # unquantized path, which is what the checkpoint stores them as.
            if isinstance(layer, LinearBase) and self._is_quantized(prefix):
                return NNCFLinearMethod(self)
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

            if input_size_per_partition % packed_factor:
                raise ValueError(
                    f"input_size_per_partition {input_size_per_partition} does "
                    f"not divide into {config.bits}-bit groups of "
                    f"{packed_factor}")
            storage = getattr(torch, _STORAGE[config.bits])
            data = torch.empty(output_size_per_partition,
                               input_size_per_partition // packed_factor,
                               dtype=storage)

            if config.bits == 4:
                # packed_dim=1: the nibble pairs run along the input dim, so a
                # row-parallel shard has to halve its offsets -- which is what
                # PackedvLLMParameter does and a plain parameter would not.
                qweight = PackedvLLMParameter(
                    data=data, input_dim=1, output_dim=0, packed_dim=1,
                    packed_factor=packed_factor, weight_loader=weight_loader)
            else:
                qweight = ModelWeightParameter(
                    data=data, input_dim=1, output_dim=0,
                    weight_loader=weight_loader)

            # The zero point has exactly the scale's shape -- one per group, or
            # one per output channel -- so it is built by the same factory and
            # shards and concatenates down the same path. Only the dtype
            # differs: it is a quantized value, not a float.
            def _per_group(dtype):
                if config.group_size > 0:
                    if input_size_per_partition % config.group_size:
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
                # Stored unpacked, one byte per group, even at 4 bits. Packing
                # it would buy ~3% of the weight bytes and cost the guarantee
                # that a fused layer can be concatenated along output channels
                # -- nibble-packing along the group dim only concatenates
                # exactly when every row holds a whole number of bytes.
                layer.register_parameter("zero_point", _per_group(torch.uint8))
            # Not params_dtype at apply time: the layer is the only thing that
            # knows what vLLM resolved --dtype to for this model.
            layer.nncf_out_dtype = params_dtype

        def process_weights_after_loading(self, layer):
            # Deliberately empty. The unquantized CPU path swaps in a oneDNN
            # GEMM here, which would consume `weight` -- a tensor this method
            # never creates -- and take the decode out of the traced graph.
            return

        def apply(self, layer, x, bias=None):
            config = self.quant_config
            weight = _dequantize(
                layer.qweight, layer.scale, config.bits, config.group_size,
                layer.nncf_out_dtype,
                zero_point=None if config.sym else layer.zero_point)
            return F.linear(x, weight, bias)

    return NNCFConfig


def register():
    """Add the method to vLLM's registry. Inert unless a checkpoint asks for it.

    Safe to call unconditionally: registering only teaches vLLM a name, and
    nothing changes for a model whose config.json does not request it.
    """
    try:
        config_cls = _build_classes()
    except Exception as _e:
        logger.debug("[OV plugin] quantization method not registered: %s", _e)
        return None
    logger.debug("[OV plugin] registered quantization method '%s'",
                 config_cls.get_name())
    return config_cls
