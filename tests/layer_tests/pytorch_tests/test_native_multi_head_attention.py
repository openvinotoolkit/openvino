# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

EMBED_DIM = 8
NUM_HEADS = 4
SEQ_LENGTH = 6
BATCH_SIZE = 1

NO_MASK, ATTN_MASK, KEY_PAD_MASK, MERGED_MASK = -1, 0, 1, 2

class aten_native_multi_head_attention(torch.nn.Module):
    def __init__(self, mask, need_weights, average_attn_weights) -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(EMBED_DIM, 3 * EMBED_DIM, dtype = torch.float32)
        self.qkv.requires_grad_(False)
        self.proj = torch.nn.Linear(EMBED_DIM, EMBED_DIM, dtype = torch.float32)
        self.proj.requires_grad_(False)

        self.embed_dim = EMBED_DIM
        self.num_heads = NUM_HEADS
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

        # Currently only int masks are working correctly, they are converted to bool.
        # Float masks raise a warning in PyTorch and are (incorrectly) converted to bool,
        # which later returns NaNs as MHA's output
        if mask == 0:
            self.mask = torch.from_numpy(np.random.randint(0, 2, (SEQ_LENGTH, SEQ_LENGTH)).astype("bool")) 
            self.mask_type = 0
        elif mask == 1:
            self.mask = torch.from_numpy(np.random.randint(0, 2, (BATCH_SIZE, SEQ_LENGTH)).astype("bool"))
            self.mask_type = 1
        elif mask == 2:
            self.mask = torch.from_numpy(np.random.randint(0, 2, (BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, SEQ_LENGTH)).astype("bool"))
            self.mask_type = 2
        else:
            self.mask = None
            self.mask_type = None

        print(self.mask)

    def forward(self, query, key, value):
        return torch.ops.aten._native_multi_head_attention(
            query, key, value, 
            embed_dim=self.embed_dim, num_head=self.num_heads,
            qkv_weight=self.qkv.weight, qkv_bias=self.qkv.bias,
            proj_weight=self.proj.weight, proj_bias=self.proj.bias,
            mask = self.mask, need_weights=self.need_weights, 
            average_attn_weights = self.average_attn_weights, 
            mask_type = self.mask_type
        )[0]

class TestNativeMultiHeadAttention(PytorchLayerTest):
    def _prepare_input(self):
        # NativeMHA is self-attention
        qkv_tensor = np.random.randn(BATCH_SIZE, SEQ_LENGTH, EMBED_DIM).astype(np.float32)
        return (qkv_tensor.copy(),
                qkv_tensor.copy(),
                qkv_tensor.copy())

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "mask", 
        [NO_MASK, ATTN_MASK, KEY_PAD_MASK, MERGED_MASK]
    )
    @pytest.mark.parametrize(
        ["need_weights", "average_attn_weights"], 
        [[False, False], [True, False], [True, True]]
    )
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122715')
    def test_native_multi_head_attention(self, ie_device, precision, ir_version, mask, need_weights, average_attn_weights):
        self._test(aten_native_multi_head_attention(mask, need_weights, average_attn_weights), 
                   None, "aten::_native_multi_head_attention", ie_device, precision, ir_version) 
