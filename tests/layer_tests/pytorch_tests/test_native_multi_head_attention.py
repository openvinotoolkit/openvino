# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

EMBED_DIM = 8
NUM_HEADS = 4
SEQ_LENGTH = 6
BATCH_SIZE = 1

NO_MASK, ATTN_MASK, KEY_PAD_MASK, MERGED_MASK = -1, 0, 1, 2

class aten_native_multi_head_attention(torch.nn.Module):
    def __init__(self, mask, need_weights, average_attn_weights, mask_data=None, mask_dtype="bool") -> None:
        super().__init__()
        self.qkv = torch.nn.Linear(EMBED_DIM, 3 * EMBED_DIM, dtype = torch.float32)
        self.qkv.requires_grad_(False)
        self.proj = torch.nn.Linear(EMBED_DIM, EMBED_DIM, dtype = torch.float32)
        self.proj.requires_grad_(False)

        self.embed_dim = EMBED_DIM
        self.num_heads = NUM_HEADS
        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights

        def make_mask(data):
            bool_mask = torch.from_numpy(data.astype("bool"))
            if mask_dtype == "float":
                # A non-boolean mask is additive: masked positions carry -inf, unmasked ones 0.
                return torch.zeros(bool_mask.shape, dtype=torch.float32).masked_fill(bool_mask, float("-inf"))
            return bool_mask

        if mask == ATTN_MASK:
            self.mask = make_mask(mask_data) if mask_data is not None else None
            self.mask_type = ATTN_MASK
        elif mask == KEY_PAD_MASK:
            self.mask = make_mask(mask_data) if mask_data is not None else None
            self.mask_type = KEY_PAD_MASK
        elif mask == MERGED_MASK:
            self.mask = make_mask(mask_data) if mask_data is not None else None
            self.mask_type = MERGED_MASK
        else:
            self.mask = None
            self.mask_type = NO_MASK

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


class aten_native_multi_head_attention_with_weights(aten_native_multi_head_attention):
    """Like aten_native_multi_head_attention, but also returns the attention weights output.
    Only meant to be used with need_weights=True, where the second output is a Tensor (not None),
    so the traced/scripted graph has a single, well defined output type."""

    def forward(self, query, key, value):
        return torch.ops.aten._native_multi_head_attention(
            query, key, value,
            embed_dim=self.embed_dim, num_head=self.num_heads,
            qkv_weight=self.qkv.weight, qkv_bias=self.qkv.bias,
            proj_weight=self.proj.weight, proj_bias=self.proj.bias,
            mask = self.mask, need_weights=self.need_weights,
            average_attn_weights = self.average_attn_weights,
            mask_type = self.mask_type
        )

class TestNativeMultiHeadAttention(PytorchLayerTest):
    def _prepare_input(self):
        # NativeMHA is self-attention
        qkv_tensor = self.random.randn(BATCH_SIZE, SEQ_LENGTH, EMBED_DIM)
        return (qkv_tensor.copy(),
                qkv_tensor.copy(),
                qkv_tensor.copy())

    def _get_mask_data(self, mask):
        """Generate mask data based on mask type."""
        if mask == ATTN_MASK:
            return self.random.randint(0, 2, (SEQ_LENGTH, SEQ_LENGTH))
        elif mask == KEY_PAD_MASK:
            return self.random.randint(0, 2, (BATCH_SIZE, SEQ_LENGTH))
        elif mask == MERGED_MASK:
            return self.random.randint(0, 2, (BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, SEQ_LENGTH))
        return None

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
        mask_data = self._get_mask_data(mask)
        self._test(aten_native_multi_head_attention(mask, need_weights, average_attn_weights, mask_data), "aten::_native_multi_head_attention", ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "mask",
        [NO_MASK, ATTN_MASK, KEY_PAD_MASK, MERGED_MASK]
    )
    @pytest.mark.parametrize("average_attn_weights", [False, True])
    @pytest.mark.parametrize("mask_dtype", ["bool", "float"])
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122715')
    def test_native_multi_head_attention_weights(self, ie_device, precision, ir_version, mask, average_attn_weights,
                                                  mask_dtype):
        # The attention weights (second output) are only produced/checked here, both averaged and
        # non-averaged. This also covers non-boolean (additive) masks: aten::_native_multi_head_attention
        # accepts them the same way as aten::_transformer_encoder_layer_fwd does.
        if ie_device == "CPU" and mask == KEY_PAD_MASK and mask_dtype == "float":
            # A key-padding mask, once unsqueezed to [batch, 1, 1, seq], is additive and has exactly
            # `seq` elements, i.e. the same size as the last dimension of the preceding QK^T MatMul.
            # The CPU plugin's post-ops fusion (DnnlPostOpsComposer::appendBinary) misidentifies this
            # Add as a fusable per-output-channel bias and fails with
            # "Check 'data.size() == OC' failed ... data size: 6 OC: 0" at graph compilation time,
            # because MatMul (unlike Convolution/FullyConnected) has no well-defined output-channel
            # axis for this fusion. This is a CPU plugin defect, not a PyTorch FE conversion issue -
            # the produced graph is mathematically correct and matches PyTorch's output.
            pytest.xfail("CPU plugin fails to compile Add(MatMul_output, mask) when the additive mask "
                         "has as many elements as the MatMul's last dimension "
                         "(dnnl_postops_composer.cpp:483, 'data.size() == OC' assertion)")
        mask_data = self._get_mask_data(mask)
        self._test(aten_native_multi_head_attention_with_weights(mask, True, average_attn_weights, mask_data,
                                                                  mask_dtype=mask_dtype),
                   "aten::_native_multi_head_attention", ie_device, precision, ir_version)
