# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestScaledDotProductAttention(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(1, 2, 8, 4).astype(np.float32), np.random.randn(1, 2, 8, 4).astype(np.float32), np.random.randn(1, 2, 8, 4).astype(np.float32))

    def create_model(self, mask, is_causal):
        import torch.nn.functional as F
        import torch

        class aten_scaled_dot_product_atten(torch.nn.Module):

            def __init__(self, mask=False, is_causal=False) -> None:
                super().__init__()

                self.mask = None if not mask else torch.from_numpy(np.random.randint(0, 2, (8, 8)).astype(np.float32))
                self.is_causal = is_causal
                if is_causal and mask:
                    self.mask.to(torch.bool)
                    self.is_causal = False

            def forward(self, query, key, value):
                return F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal)

        ref_net = None

        return aten_scaled_dot_product_atten(mask, is_causal), ref_net, "aten::scaled_dot_product_attention"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize(['mask', "is_causal"], [(False, False), (False, True), (True, True), (True, False)])
    def test_scaled_dot_product_atten(self, ie_device, precision, ir_version, mask, is_causal):
        self._test(*self.create_model(mask, is_causal),ie_device, precision, ir_version)
