# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

EMBED_DIM = 8
NUM_HEADS = 4
FFN_DIM = 16
SEQ_LENGTH = 6
BATCH_SIZE = 2

NO_MASK, ATTN_MASK, KEY_PAD_MASK, MERGED_MASK = -1, 0, 1, 2


def make_encoder_layer(norm_first, activation):
    layer = torch.nn.TransformerEncoderLayer(d_model=EMBED_DIM,
                                             nhead=NUM_HEADS,
                                             dim_feedforward=FFN_DIM,
                                             dropout=0.0,
                                             activation=activation,
                                             norm_first=norm_first,
                                             batch_first=True,
                                             dtype=torch.float32)
    layer.eval()
    layer.requires_grad_(False)
    return layer


class aten_transformer_encoder_layer_fwd(torch.nn.Module):
    """Calls the fast-path operation directly, so that it is present both in
    TorchScript and in the graph produced by torch.export."""

    def __init__(self, norm_first, use_gelu, mask_type, mask_data=None) -> None:
        super().__init__()
        self.layer = make_encoder_layer(norm_first, "gelu" if use_gelu else "relu")
        self.use_gelu = use_gelu
        self.norm_first = norm_first
        self.mask_type = mask_type if mask_type != NO_MASK else None
        # PyTorch canonicalizes boolean masks into additive float masks before
        # calling the fast path operation.
        self.mask = None
        if mask_data is not None:
            self.mask = torch.zeros(mask_data.shape, dtype=torch.float32).masked_fill(
                torch.from_numpy(mask_data.astype("bool")), float("-inf"))

    def forward(self, src):
        layer = self.layer
        return torch.ops.aten._transformer_encoder_layer_fwd(
            src, EMBED_DIM, NUM_HEADS,
            layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias,
            layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias,
            self.use_gelu, self.norm_first, layer.norm1.eps,
            layer.norm1.weight, layer.norm1.bias,
            layer.norm2.weight, layer.norm2.bias,
            layer.linear1.weight, layer.linear1.bias,
            layer.linear2.weight, layer.linear2.bias,
            self.mask, self.mask_type)


class aten_transformer_encoder_layer_module(torch.nn.Module):
    """Uses the public module, which dispatches to the fast path when traced."""

    def __init__(self, norm_first, activation, mask_type=NO_MASK, mask_data=None) -> None:
        super().__init__()
        self.layer = make_encoder_layer(norm_first, activation)
        self.mask_type = mask_type
        self.mask = torch.from_numpy(mask_data.astype("bool")) if mask_data is not None else None

    def forward(self, src):
        if self.mask_type == KEY_PAD_MASK:
            return self.layer(src, src_key_padding_mask=self.mask)
        if self.mask_type == ATTN_MASK:
            return self.layer(src, src_mask=self.mask)
        return self.layer(src)


class TestTransformerEncoderLayerFwd(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(BATCH_SIZE, SEQ_LENGTH, EMBED_DIM).astype("float32"),)

    def _get_mask_data(self, mask_type):
        if mask_type == ATTN_MASK:
            # Keep at least one unmasked position per row to avoid NaNs in softmax
            mask = self.random.randint(0, 2, (SEQ_LENGTH, SEQ_LENGTH))
            mask[:, 0] = 0
            return mask
        if mask_type == KEY_PAD_MASK:
            # Keep at least one unmasked position per row to avoid NaNs in softmax
            mask = self.random.randint(0, 2, (BATCH_SIZE, SEQ_LENGTH))
            mask[:, 0] = 0
            return mask
        if mask_type == MERGED_MASK:
            mask = self.random.randint(0, 2, (BATCH_SIZE, NUM_HEADS, SEQ_LENGTH, SEQ_LENGTH))
            mask[:, :, :, 0] = 0
            return mask
        return None

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("norm_first", [False, True])
    @pytest.mark.parametrize("use_gelu", [False, True])
    @pytest.mark.parametrize("mask_type", [NO_MASK, ATTN_MASK, KEY_PAD_MASK, MERGED_MASK])
    def test_transformer_encoder_layer_fwd(self, norm_first, use_gelu, mask_type,
                                           ie_device, precision, ir_version):
        mask_data = self._get_mask_data(mask_type)
        self._test(aten_transformer_encoder_layer_fwd(norm_first, use_gelu, mask_type, mask_data),
                   "aten::_transformer_encoder_layer_fwd", ie_device, precision, ir_version,
                   trace_model=True, custom_eps=1e-4)

    @pytest.mark.nightly
    @pytest.mark.precommit
    # norm_first=True disables PyTorch's fused fast path (why_not_sparsity_fast_path =
    # "norm_first was True"), so the traced graph decomposes into elementary ops and
    # aten::_transformer_encoder_layer_fwd is never emitted. That case is already covered
    # by test_transformer_encoder_layer_fwd, which calls the fast-path op directly.
    @pytest.mark.parametrize("norm_first", [False])
    @pytest.mark.parametrize("activation", ["relu", "gelu"])
    @pytest.mark.skipif(PytorchLayerTest.use_torch_export(),
                        reason="TransformerEncoderLayer fast path is not used by torch.export")
    @pytest.mark.parametrize("mask_type", [NO_MASK, ATTN_MASK, KEY_PAD_MASK])
    def test_transformer_encoder_layer_module(self, norm_first, activation, mask_type,
                                              ie_device, precision, ir_version):
        mask_data = self._get_mask_data(mask_type)
        self._test(aten_transformer_encoder_layer_module(norm_first, activation, mask_type, mask_data),
                   "aten::_transformer_encoder_layer_fwd", ie_device, precision, ir_version,
                   trace_model=True, custom_eps=1e-4)
