# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestScaledDotProductAttention(PytorchLayerTest):

    def _prepare_input(self, dtype):
        return (np.random.randn(1, 2, 8, 4).astype(dtype), np.random.randn(1, 2, 8, 4).astype(dtype), np.random.randn(1, 2, 8, 4).astype(dtype))

    def create_model(self, mask, is_causal, scale, dtype):
        import torch.nn.functional as F
        import torch

        class aten_scaled_dot_product_atten(torch.nn.Module):

            def __init__(self, mask=False, is_causal=False, scale=False, dtype=np.float32) -> None:
                super().__init__()

                self.mask = None if not mask else torch.from_numpy(
                    np.random.randint(0, 2, (8, 8)).astype(dtype))
                self.is_causal = is_causal
                if is_causal and mask:
                    self.mask.to(torch.bool)
                    self.is_causal = False

                self.scale = None if not scale else torch.tensor(
                    5, dtype=torch.float)

            def forward(self, query, key, value):
                return F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal, scale=self.scale)

        ref_net = None

        return aten_scaled_dot_product_atten(mask, is_causal, scale, dtype), ref_net, 'aten::scaled_dot_product_attention'

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize(['mask', 'is_causal'], [(False, False), (False, True), (True, True), (True, False)])
    @pytest.mark.parametrize('scale', [False, True])
    @pytest.mark.parametrize("dtype", (np.float32, np.float64))
    def test_scaled_dot_product_atten(self, ie_device, precision, ir_version, mask, is_causal, scale, dtype):
        self._test(*self.create_model(mask, is_causal, scale, dtype),
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"dtype": dtype})
