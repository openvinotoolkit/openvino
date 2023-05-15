# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

EMBED_DIM = 8
NUM_HEADS = 4
SEQ_LENGTH = 4
BATCH_SIZE = 1

ATTN_MASK, KEY_PAD_MASK, MERGED_MASK = 0, 1, 2

class aten_native_multi_head_attention(torch.nn.Module):
    def __init__(self, mask, need_weights, average_attn_weights) -> None:
        super().__init__()
        self.qkv_weight = torch.from_numpy(np.random.randn(3 * EMBED_DIM, EMBED_DIM).astype(np.float32))
        self.qkv_bias = torch.from_numpy(np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32))
        self.proj_weight = torch.from_numpy(np.random.randn(EMBED_DIM, EMBED_DIM).astype(np.float32))
        self.proj_bias = torch.from_numpy(np.random.randn(EMBED_DIM).astype(np.float32))
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

        if mask == 0:
            self.mask = torch.from_numpy(np.random.randn(SEQ_LENGTH, SEQ_LENGTH).astype(np.float32))
            self.mask_type = 0
        elif mask == 1:
            self.mask = torch.from_numpy(np.random.randn(BATCH_SIZE, SEQ_LENGTH).astype(np.float32))
            self.mask_type = 1
        elif mask == 2:
            self.mask = torch.from_numpy(np.random.randn(BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, SEQ_LENGTH).astype(np.float32))
            self.mask_type = 2
        else:
            self.mask = None
            self.mask_type = None

    def forward(self, query, key, value):
        return torch._native_multi_head_attention(
            query, key, value, 
            embed_dim=EMBED_DIM, num_head=NUM_HEADS,
            qkv_weight=self.qkv_weight, qkv_bias=self.qkv_bias,
            proj_weight=self.proj_weight, proj_bias=self.proj_bias,
            mask = self.mask, mask_type=self.mask_type,
            need_weights=self.need_weights, average_attn_weights=self.average_attn_weights
        )

class TestNativeMultiHeadAttention(PytorchLayerTest):
    def _prepare_input(self):
        # NativeMHA is self-attention
        qkv_tensor = np.random.randn(BATCH_SIZE, SEQ_LENGTH, EMBED_DIM).astype(np.float32)
        return (qkv_tensor,
                qkv_tensor,
                qkv_tensor)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "mask", 
        [None, ATTN_MASK, KEY_PAD_MASK, MERGED_MASK]
    )
    @pytest.mark.parametrize(
        "need_weights", 
        [False, True]
    )
    @pytest.mark.parametrize(
        "average_attn_weights", 
        [False, True]
    )
    def test_native_multi_head_attention(self, ie_device, precision, ir_version, mask, need_weights, average_attn_weights):
        self._test(aten_native_multi_head_attention(mask, need_weights, average_attn_weights), 
                   None, "aten::_native_multi_head_attention", ie_device, precision, ir_version)
