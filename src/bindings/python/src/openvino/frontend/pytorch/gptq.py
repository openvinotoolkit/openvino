# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from functools import partial
import logging
import torch

log = logging.getLogger(__name__)


# Wraps a single tensor to a module to prevent it from jit.freezing
# It depends on a tensor dtype whether it will be preserved from freezing.
# Refer to the decoder code to learn which types will be preserved.
class KeepWeight(torch.nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self):
        return self.weight


# Produces a pattern that can be captured later and represented as a single u4 constant node
def decompression_pattern(weights):
    mask = torch.tensor(15, dtype=torch.uint8).to(weights.device)
    return torch.stack((torch.bitwise_and(weights, mask), torch.bitwise_right_shift(weights, 4)), dim=-1)


def patched_forward(self, *args, **kwargs):
    if hasattr(self, "_hf_hook"):
        args, kwargs = self._hf_hook.pre_forward(self, *args, **kwargs)

    arg = args[0]
    dtype = arg.dtype
    outshape = arg.shape[:-1] + (self.width,)
    arg = arg.contiguous().view(-1, arg.shape[-1])
    groups = self.qzeros.shape[0]
    height = self.qweight.shape[0]

    unpacked_weights = decompression_pattern(
        self._openvino_u4_compression_submodule_qweights()).contiguous().view(height, -1, 8)
    unpacked_weights = torch.transpose(
        unpacked_weights, 1, 2).contiguous().view(-1, self.group_size, self.width)
    unpacked_zp = decompression_pattern(
        self._openvino_u4_compression_submodule_qzeros()).contiguous().view(groups, 1, -1)

    unpacked_weights = (unpacked_weights.to(dtype) - unpacked_zp) * self.scales
    unpacked_weights = unpacked_weights.view(-1, self.width)

    out = arg @ unpacked_weights

    out = out.view(outshape)
    if self.bias is not None:
        out.add_(self.bias)

    if hasattr(self, "_hf_hook"):
        out = self._hf_hook.post_forward(self, out)
    return out


def patched_forward_sym(self, *args, **kwargs):
    if hasattr(self, "_hf_hook"):
        args, kwargs = self._hf_hook.pre_forward(self, *args, **kwargs)

    arg = args[0]
    dtype = arg.dtype
    outshape = arg.shape[:-1] + (self.width,)
    arg = arg.contiguous().view(-1, arg.shape[-1])
    height = self.qweight.shape[0]

    unpacked_weights = decompression_pattern(
        self._openvino_u4_compression_submodule_qweights()).contiguous().view(height, -1, 8)
    unpacked_weights = torch.transpose(
        unpacked_weights, 1, 2).contiguous().view(-1, self.group_size, self.width)

    # all zp is 8 for symmetrical, will repack to i4 in pt fe transformation
    unpacked_weights = (unpacked_weights.to(torch.int8) - torch.tensor(8, dtype=torch.int8))
    unpacked_weights = unpacked_weights.to(dtype) * self.scales
    unpacked_weights = unpacked_weights.view(-1, self.width)

    out = arg @ unpacked_weights

    out = out.view(outshape)
    if self.bias is not None:
        out.add_(self.bias)

    if hasattr(self, "_hf_hook"):
        out = self._hf_hook.post_forward(self, out)
    return out


# All the following AutoGPTQ"s quant types are supposed to have the same weights packing schema
supported_quant_types = ["triton", "exllama", "exllamav2", "cuda-old"]


def patch_model(model):
    is_symmetrical = False
    config = None
    if hasattr(model, "config"):
        config = model.config
    elif hasattr(model, "model") and hasattr(model.model, "config"):
        # original model was wrapped
        config = model.model.config
    if config is not None and hasattr(config, "quantization_config") and hasattr(config.quantization_config, "sym"):
        is_symmetrical = config.quantization_config.sym
    for name, module in model.named_modules():
        if hasattr(module, "_openvino_patch_orig_forward"):
            # already patched, skipping
            continue
        # TODO: Check module type
        is_quantized = getattr(module, "is_quantized", None)
        if is_quantized is not None:
            module.is_quantized = False
        module.float()  # enables tracing on CPU, applied for all modules
        if hasattr(module, "QUANT_TYPE"):
            if module.QUANT_TYPE not in supported_quant_types:
                raise ValueError(f"Unsupported QUANT_TYPE == {module.QUANT_TYPE} is discovered for "
                                 "AutoGPTQ model, only the following types are supported: "
                                 f"{supported_quant_types}")
            if module.bits != 4:
                raise ValueError(f"Unsupported bits == {module.bits} is discovered in module {name} "
                                 "in AutoGPTQ model, only bits == 4 is supported.")

            int4_in_int32 = 8
            groups = module.qzeros.shape[0]
            module.width = module.qweight.shape[1]
            assert module.group_size == module.qweight.shape[0] * int4_in_int32 // groups

            module._openvino_patch_orig_forward = module.forward
            if is_symmetrical:
                module.forward = partial(patched_forward_sym, module)
            else:
                module.forward = partial(patched_forward, module)

            # Keep original field properties to be used when model is returned back to its original state
            module._openvino_patch_orig_qweights_type = module.qweight.dtype
            module._openvino_patch_orig_qzeros_type = module.qzeros.dtype
            module._openvino_patch_orig_scale_shape = module.scales.shape

            module.qweight = module.qweight.view(dtype=torch.uint8)
            module.qzeros = module.qzeros.view(dtype=torch.uint8)

            # TODO: Redundant tensor copy? Try to remove module.qweight and module.qzeros after keeping modified values as submodules
            module.add_module(
                "_openvino_u4_compression_submodule_qweights", KeepWeight(module.qweight))
            # Adding 17 to move zp+1 step from after unpacking to before to have correct decompression pattern. Can it overflow?
            module.add_module("_openvino_u4_compression_submodule_qzeros",
                              KeepWeight(module.qzeros + torch.tensor(17, dtype=torch.uint8)))

            module.scales = module.scales.view(-1, 1, module.width)


def unpatch_model(model):
    for _, module in model.named_modules():
        if hasattr(module, "_openvino_patch_orig_forward"):
            try:
                module.forward = module._openvino_patch_orig_forward
                del module._openvino_patch_orig_forward

                module.qweight = module.qweight.view(
                    dtype=module._openvino_patch_orig_qweights_type)
                del module._openvino_patch_orig_qweights_type

                module.qzeros = module.qzeros.view(
                    dtype=module._openvino_patch_orig_qzeros_type)
                del module._openvino_patch_orig_qzeros_type

                module.scales = module.scales.view(module._openvino_patch_orig_scale_shape)
                del module._openvino_patch_orig_scale_shape

                del module._openvino_u4_compression_submodule_qweights
                del module._openvino_u4_compression_submodule_qzeros
            except Exception as error:
                log.warning("Exception raised during GPTQ model unpatching. "
                            "Depending on the exact issue it may lead to broken "
                            "original model.\n%s", error)
