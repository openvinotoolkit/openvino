# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


"""vLLM `"quant_method": "nncf"`: NNCF weight-only int4/int8 (sym or asym, grouped or per-channel), decoded in torch.

The classes are built lazily (keeping torch and vLLM out of import) and resolved by the module `__getattr__`, so an
NNCFConfig pickles into a spawned engine process.
"""

import logging
import threading
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# In the order `_define_classes` returns them.
_LAZY_CLASSES = ("NNCFConfig", "NNCFLinearMethod", "NNCFFusedMoEMethod")
_classes: tuple[type[Any], type[Any], type[Any]] | None = None
# Not re-entrant: `_define_classes` reaches NNCFLinearMethod through a closure, not this module.
_classes_lock = threading.Lock()

# Storage dtype per `bits`; 4-bit is packed two per byte.
_STORAGE = {4: "uint8", 8: "int8"}


def _unpack_uint4(packed: "torch.Tensor") -> "torch.Tensor":
    """Asymmetric int4 -> uint8 in [0, 15], low nibble first, gaining a trailing dim of 2."""
    import torch

    low = torch.bitwise_and(packed, 15)
    high = torch.bitwise_right_shift(packed, 4)
    return torch.stack((low, high), dim=-1)


def _unpack_int4(packed: "torch.Tensor") -> "torch.Tensor":
    """Symmetric int4 -> int8 in [-8, 7]; the -8 undoes the unsigned bias."""
    import torch

    return _unpack_uint4(packed).type(torch.int8) - 8


def _dequantize(qweight: "torch.Tensor", scale: "torch.Tensor", bits: int, out_dtype: "torch.dtype",
                zero_point: "torch.Tensor | None" = None) -> "torch.Tensor":
    """Decompress to a dense [..., out, in] weight in `out_dtype`, in NNCF's order so it matches bit for bit.

    Leading dims are MoE experts. scale and zero_point are [..., out, groups], or [..., out, groups, 1] already
    broadcastable: OV only marks the chain as decompression, and keeps it compressed into a (Gather)MatMul, when
    they reach the Subtract/Multiply as constants, not through an Unsqueeze.
    """
    rows = qweight.shape[:-1]
    if bits == 4:
        in_features = qweight.shape[-1] * 2
        weight = (_unpack_int4(qweight) if zero_point is None
                  else _unpack_uint4(qweight))
    else:
        in_features = qweight.shape[-1]
        weight = qweight

    # groups == 1 is per output channel.
    expand = scale.dim() == qweight.dim()
    groups = scale.shape[-1 if expand else -2]
    weight = weight.reshape(*rows, groups, in_features // groups)
    weight = weight.type(scale.dtype)
    if zero_point is not None:
        zero_point = zero_point.type(scale.dtype)
        weight = weight - (zero_point.unsqueeze(-1) if expand else zero_point)
    weight = weight * (scale.unsqueeze(-1) if expand else scale)
    return weight.reshape(*rows, in_features).type(out_dtype)


def _define_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """Define the config and its methods; call `_build_classes` instead, which builds them only once."""
    import torch
    from torch.nn import functional
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import FusedMoEMethodBase
    from vllm.model_executor.layers.fused_moe.routed_experts import FusedMoeWeightScaleSupported, RoutedExperts
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
    from vllm.model_executor.utils import set_weight_attrs

    @register_quantization_config("nncf")
    class NNCFConfig(QuantizationConfig):
        """Weight-only integer compression, as written by NNCF."""

        def __init__(self, bits: int, group_size: int, sym: bool, scale_dtype: str,
                     quantized_modules: Iterable[str] | None) -> None:
            super().__init__()
            if bits not in _STORAGE:
                raise ValueError(f"bits={bits} is not one of {sorted(_STORAGE)}")
            self.bits = bits
            self.sym = sym
            self.group_size = group_size
            self.scale_dtype = getattr(torch, scale_dtype)
            # Empty means every linear layer.
            self.quantized_modules = set(quantized_modules or ())

        def __repr__(self) -> str:
            return (f"NNCFConfig(bits={self.bits}, "
                    f"{'sym' if self.sym else 'asym'}, "
                    f"group_size={self.group_size}, "
                    f"scale_dtype={self.scale_dtype}, "
                    f"modules={len(self.quantized_modules) or 'all'})")

        @classmethod
        def get_name(cls) -> str:
            return "nncf"

        @classmethod
        def get_supported_act_dtypes(cls) -> list[torch.dtype]:
            return [torch.float32, torch.float16, torch.bfloat16]

        @classmethod
        def get_min_capability(cls) -> int:
            return 60

        @staticmethod
        def get_config_filenames() -> list[str]:
            return []

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "NNCFConfig":
            return cls(
                bits=config.get("bits", 4),
                group_size=config.get("group_size", -1),
                sym=config.get("sym", True),
                scale_dtype=config.get("scale_dtype", "float16"),
                quantized_modules=config.get("quantized_modules", ()),
            )

        def apply_vllm_mapper(self, hf_to_vllm_mapper: Any) -> None:
            # `_is_quantized` gets vLLM prefixes; unmapped names would leave renamed models silently dense.
            self.quantized_modules = set(
                hf_to_vllm_mapper.apply_list(sorted(self.quantized_modules)))

        def _is_quantized(self, prefix: str) -> bool:
            """Is the vLLM layer at `prefix` compressed? A fused layer only if all of its sources are."""
            if not self.quantized_modules:
                return True
            parent, _, leaf = prefix.rpartition(".")
            sources = self.packed_modules_mapping.get(leaf)
            names = ([f"{parent}.{s}" for s in sources] if sources else [prefix])
            return all(n in self.quantized_modules for n in names)

        def get_quant_method(self, layer: torch.nn.Module,
                             prefix: str) -> "LinearMethodBase | FusedMoEMethodBase | None":
            # vLLM rejects None for a linear layer, so a dense one gets Unquantized.
            if isinstance(layer, LinearBase):
                if self._is_quantized(prefix):
                    return NNCFLinearMethod(self)
                return UnquantizedLinearMethod()
            # Experts are listed by their FusedMoE module (`...mlp.experts`); None leaves them dense.
            if isinstance(layer, RoutedExperts) and self._is_quantized(prefix):
                return NNCFFusedMoEMethod(self, layer.moe_config)
            return None

    class NNCFLinearMethod(LinearMethodBase):
        """Declares the compressed parameters, then decodes them per forward."""

        def __init__(self, quant_config: NNCFConfig) -> None:
            self.quant_config = quant_config

        def create_weights(self, layer: torch.nn.Module, input_size_per_partition: int,
                           output_partition_sizes: list[int], input_size: int, output_size: int,
                           params_dtype: torch.dtype, **extra_weight_attrs: Any) -> None:
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
                # Nibble pairs run along the input dim, so a row-parallel shard halves its offsets.
                qweight = PackedvLLMParameter(
                    data=data, input_dim=1, output_dim=0, packed_dim=1,
                    packed_factor=packed_factor, weight_loader=weight_loader)
            else:
                qweight = ModelWeightParameter(
                    data=data, input_dim=1, output_dim=0,
                    weight_loader=weight_loader)

            # Also used for the zero point, so it shards and concatenates like the scale.
            def _per_group(dtype: torch.dtype) -> "GroupQuantScaleParameter | ChannelQuantScaleParameter":
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
                # Unpacked: packing would break output-dim concatenation of fused layers.
                layer.register_parameter("zero_point", _per_group(torch.uint8))
            layer.nncf_out_dtype = params_dtype

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # Deliberately empty: the unquantized CPU path would swap in a oneDNN GEMM over a missing `weight`.
            return

        def apply(self, layer: torch.nn.Module, inputs: torch.Tensor,
                  bias: torch.Tensor | None = None) -> torch.Tensor:
            config = self.quant_config
            weight = _dequantize(
                layer.qweight, layer.scale, config.bits,
                layer.nncf_out_dtype,
                zero_point=None if config.sym else layer.zero_point)
            return functional.linear(inputs, weight, bias)

    class NNCFFusedMoEMethod(FusedMoEMethodBase):
        """Routed experts kept int4 asym, decoded by `fused_moe.nncf_moe_forward`.

        Monolithic, so routing is vLLM's torch `select_experts` (the modular router calls `_moe_C.topk_softmax`,
        which OV cannot take). Loads into fused `w13_*`, then splits gate and up: a slice between a constant
        and its decompression would hide the compressed weight from OV.
        """

        _FIELDS = ("qweight", "scale", "zero_point")

        def __init__(self, quant_config: NNCFConfig, moe: Any) -> None:
            super().__init__(moe)
            self.quant_config = quant_config

        @property
        def is_monolithic(self) -> bool:
            return True

        def create_weights(self, layer: torch.nn.Module, num_experts: int, hidden_size: int,
                           intermediate_size_per_partition: int, params_dtype: torch.dtype,
                           **extra_weight_attrs: Any) -> None:
            config = self.quant_config
            if config.bits != 4 or config.sym or config.group_size <= 0:
                raise ValueError(f"[OV plugin] nncf MoE experts need grouped int4 asym, got {config}")
            if layer.activation.value not in ("silu", "gelu", "gelu_tanh"):
                raise ValueError(f"[OV plugin] nncf MoE experts: activation {layer.activation} is not supported")
            if layer.apply_router_weight_on_input:
                raise ValueError("[OV plugin] nncf MoE experts: apply_router_weight_on_input is not supported")
            # GROUP routes scale and zero_point through FusedMoE's grouped loader, sharded like the weight.
            attrs = {"weight_loader": extra_weight_attrs["weight_loader"],
                     "quant_method": FusedMoeWeightScaleSupported.GROUP.value}
            inter = intermediate_size_per_partition
            for name, rows, cols in (("w13", 2 * inter, hidden_size), ("w2", hidden_size, inter)):
                if cols % config.group_size:
                    raise ValueError(f"[OV plugin] nncf MoE {name}: {cols} is not a multiple of "
                                     f"group_size {config.group_size}")
                groups = cols // config.group_size
                # zero_point unpacked, as for linears.
                for field, shape, dtype in (("qweight", (rows, cols // 2), torch.uint8),
                                            ("scale", (rows, groups), config.scale_dtype),
                                            ("zero_point", (rows, groups), torch.uint8)):
                    param = torch.nn.Parameter(torch.empty(num_experts, *shape, dtype=dtype), requires_grad=False)
                    layer.register_parameter(f"{name}_{field}", param)
                    set_weight_attrs(param, attrs)
            layer.nncf_out_dtype = params_dtype

        def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
            # scale/zero_point gain their broadcast dim here, as constants (see _dequantize).
            for field in self._FIELDS:
                tensors = {"w2": getattr(layer, f"w2_{field}").data}
                tensors["w1"], tensors["w3"] = getattr(layer, f"w13_{field}").data.chunk(2, dim=1)
                delattr(layer, f"w13_{field}")
                for name, tensor in tensors.items():
                    if field != "qweight":
                        tensor = tensor.unsqueeze(-1)
                    layer.register_parameter(
                        f"{name}_{field}", torch.nn.Parameter(tensor.contiguous(), requires_grad=False))

        def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> None:
            return None

        def apply_monolithic(self, layer: torch.nn.Module, x: torch.Tensor, router_logits: torch.Tensor,
                             input_ids: torch.Tensor | None = None) -> torch.Tensor:
            from vllm.model_executor.layers.fused_moe.cpu_fused_moe import select_experts

            from openvino.frontend.pytorch.torchdynamo.vllm.fused_moe import nncf_moe_forward

            topk_weights, topk_ids = select_experts(
                hidden_states=x, router_logits=router_logits, top_k=layer.top_k,
                use_grouped_topk=layer.use_grouped_topk, renormalize=layer.renormalize,
                topk_group=layer.topk_group, num_expert_group=layer.num_expert_group,
                custom_routing_function=layer.custom_routing_function, scoring_func=layer.scoring_func,
                routed_scaling_factor=layer.routed_scaling_factor,
                e_score_correction_bias=layer.e_score_correction_bias)
            weights = [[getattr(layer, f"{name}_{field}") for field in self._FIELDS] for name in ("w1", "w3", "w2")]
            return nncf_moe_forward(x, weights, topk_weights, topk_ids, layer.activation.value,
                                    lambda q, s, zp: _dequantize(q, s, 4, layer.nncf_out_dtype, zero_point=zp))

    # Pickle stores module + qualname; the module `__getattr__` resolves these.
    for cls in (NNCFConfig, NNCFLinearMethod, NNCFFusedMoEMethod):
        cls.__module__ = __name__
        cls.__qualname__ = cls.__name__

    return NNCFConfig, NNCFLinearMethod, NNCFFusedMoEMethod


def _build_classes() -> tuple[type[Any], type[Any], type[Any]]:
    """`_define_classes`, memoized: pickle refuses a class whose reference resolves to another object."""
    global _classes
    if _classes is not None:
        return _classes
    with _classes_lock:
        if _classes is None:
            _classes = _define_classes()
    return _classes


def __getattr__(name: str) -> type[Any]:
    """Resolve the lazily built classes as module attributes (PEP 562), for pickle."""
    if name not in _LAZY_CLASSES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return _build_classes()[_LAZY_CLASSES.index(name)]
    except Exception as exc:
        # AttributeError, not ImportError, keeps hasattr() and getattr(..., default) working.
        raise AttributeError(
            f"module {__name__!r} cannot provide {name!r}: {exc}") from exc


def register() -> type[Any] | None:
    """Add the method to vLLM's registry; safe to call repeatedly and in any process."""
    try:
        config_cls = _build_classes()[0]
    except Exception as _e:
        logger.debug("[OV plugin] quantization method not registered: %s", _e)
        return None
    logger.debug("[OV plugin] registered quantization method '%s'",
                 config_cls.get_name())
    return config_cls
