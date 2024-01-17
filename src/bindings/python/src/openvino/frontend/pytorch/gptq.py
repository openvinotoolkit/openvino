
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import torch
from functools import partial

# Wraps a single tensor to a module to prevent it from jit.freezing
# It depends on a tensor dtype whether it will be preserved from freezing. Refer to the decoder code to learn which types will be preserved.
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
    if hasattr(self, '_hf_hook'):
        args, kwargs = self._hf_hook.pre_forward(self, *args, **kwargs)

    x = args[0]
    dtype = x.dtype
    outshape = x.shape[:-1] + (self.width,)
    x = x.view(-1, x.shape[-1])
    groups = self.qzeros.shape[0]
    height = self.qweight.shape[0]

    unpacked_weights = decompression_pattern(
        self._openvino_u4_compression_submodule_qweights()).contiguous().view(height, -1, 8)
    unpacked_weights = torch.transpose(
        unpacked_weights, 1, 2).contiguous().view(-1, self.group_size, self.width)
    unpacked_zp = decompression_pattern(
        self._openvino_u4_compression_submodule_qzeros()).contiguous().view(groups, 1, -1)

    unpacked_zp = unpacked_zp.to(dtype) + 1

    unpacked_weights = (unpacked_weights.to(dtype) - unpacked_zp) * self.scales
    unpacked_weights = unpacked_weights.view(-1, self.width)

    out = x @ unpacked_weights

    out = out.view(outshape)
    if self.bias is not None:
        out.add_(self.bias)

    if hasattr(self, '_hf_hook'):
        out = self._hf_hook.post_forward(self, out)
    return out


# All the following AutoGPTQ's quant types are supposed to have the same weights packing schema
supported_quant_types = ['triton', 'exllama', 'cuda', 'exllamav2', 'cuda-old']


def patch_model(model):
    for name, m in model.named_modules():
        if hasattr(m, '_openvino_patch_orig_forward'):
            # already patched, skipping
            continue
        # TODO: Check module type
        is_quantized = getattr(m, 'is_quantized', None)
        if is_quantized is not None:
            m.is_quantized = False
        m.float()  # enables tracing on CPU, applied for all modules
        if hasattr(m, 'QUANT_TYPE'):
            if m.QUANT_TYPE not in supported_quant_types:
                raise ValueError(
                    f'Unsupported QUANT_TYPE == {m.QUANT_TYPE} is discovered for AutoGPTQ model, only the following types are supported: {supported_quant_types}')
            if m.bits != 4:
                raise ValueError(
                    f'Unsupported bits == {m.bits} is discovered in module {name} in AutoGPTQ model, only bits == 4 is supported.')

            int4_in_int32 = 8
            groups = m.qzeros.shape[0]
            m.width = m.qweight.shape[1]
            assert m.group_size == m.qweight.shape[0] * int4_in_int32 // groups

            m._openvino_patch_orig_forward = m.forward
            m.forward = partial(patched_forward, m)

            # Keep original field properties to be used when model is returned back to its original state
            m._openvino_patch_orig_qweights_type = m.qweight.dtype
            m._openvino_patch_orig_qzeros_type = m.qzeros.dtype
            m._openvino_patch_orig_scale_shape = m.scales.shape

            m.qweight = m.qweight.view(dtype=torch.uint8)
            m.qzeros = m.qzeros.view(dtype=torch.uint8)

            # TODO: Redundant tensor copy? Try to remove m.qweigh and m.qzeros after keeping modified values as submodules
            m.add_module(
                '_openvino_u4_compression_submodule_qweights', KeepWeight(m.qweight))
            m.add_module('_openvino_u4_compression_submodule_qzeros',
                         KeepWeight(m.qzeros))

            m.scales = m.scales.view(-1, 1, m.width)


def unpatch_model(model):
    for _, m in model.named_modules():
        if hasattr(m, '_openvino_patch_orig_forward'):
            try:
                m.forward = m._openvino_patch_orig_forward
                del m._openvino_patch_orig_forward

                m.qweight = m.qweight.view(
                    dtype=m._openvino_patch_orig_qweights_type)
                del m._openvino_patch_orig_qweights_type

                m.qzeros = m.qzeros.view(
                    dtype=m._openvino_patch_orig_qzeros_type)
                del m._openvino_patch_orig_qzeros_type

                m.scales = m.scales.view(m._openvino_patch_orig_scale_shape)
                del m._openvino_patch_orig_scale_shape

                del m._openvino_u4_compression_submodule_qweights
                del m._openvino_u4_compression_submodule_qzeros
            except Exception as error:
                print('[ WARNING ] Exception raised during GPTQ model unpatching. Depending on the exact issue it may lead to broken original model')
                print(error)


def detect_gptq_model_raw(model):
    return model and getattr(model, 'config', None) and getattr(model.config, 'quantization_config', None) and model.config.quantization_config.quant_method == 'gptq'


def detect_gptq_model(model):
    return detect_gptq_model_raw(model) or getattr(model, 'model', None) and detect_gptq_model_raw(model.model)
