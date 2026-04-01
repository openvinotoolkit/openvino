# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestScaledDotProductAttention(PytorchLayerTest):
    def _prepare_input(self, dtype):
        return (self.random.randn(1, 2, 8, 4, dtype=dtype),
                self.random.randn(1, 2, 16, 4, dtype=dtype),
                self.random.randn(1, 2, 16, 6, dtype=dtype))

    def create_model(self, mask, is_causal, dtype, mask_shape, enable_gqa):
        import torch.nn.functional as F
        import torch

        # Generate mask outside inner class to use test's seeded random
        attn_mask = None if not mask else torch.from_numpy(
            self.random.randint(0, 2, mask_shape, dtype=dtype))

        class aten_sdpa(torch.nn.Module):
            def __init__(self, mask=False, is_causal=False,
                         dtype=np.float32, mask_shape=None, enable_gqa=False) -> None:
                super().__init__()
                self.mask = attn_mask
                self.is_causal = False if (is_causal and mask) else is_causal
                self.enable_gqa = enable_gqa

            def forward(self, query, key, value):
                # torch export struggles with dynamic scale
                a = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   enable_gqa=self.enable_gqa)
                b = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   scale=torch.tensor(5, dtype=torch.float), enable_gqa=self.enable_gqa)
                return a, b

        return (aten_sdpa(mask, is_causal, dtype, mask_shape, enable_gqa),
                'aten::scaled_dot_product_attention')

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize('mask', (True, False))
    @pytest.mark.parametrize('is_causal', (True, False))
    @pytest.mark.parametrize('mask_shape', [(8, 16),
                                            (1, 16),
                                            (8, 1),
                                            (1, 1),
                                            (1, 8, 1),
                                            (1, 1, 8, 1),
                                            (1, 2, 8, 16)])
    @pytest.mark.parametrize('dyn_shapes', (True, False))
    @pytest.mark.parametrize('enable_gqa', (False, True))
    def test_scaled_dot_product_atten(self, ie_device, precision, ir_version,
                                      mask, is_causal, mask_shape, dyn_shapes, enable_gqa):
        if PytorchLayerTest.use_torch_export() and not mask and is_causal:
            pytest.xfail(reason="Unsupported case for torch.export")
        dtype = np.float32
        self._test(*self.create_model(mask, is_causal, dtype, mask_shape, enable_gqa),
                   ie_device, precision, ir_version, dynamic_shapes=dyn_shapes,
                   kwargs_to_prepare_input={"dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.parametrize(['mask', 'is_causal'], [(False, False),
                                                     (False, True),
                                                     (True, True),
                                                     (True, False)])
    @pytest.mark.parametrize('mask_shape', [(8, 16),
                                            (1, 16),
                                            (8, 1),
                                            (1, 1),
                                            (1, 8, 1),
                                            (1, 1, 8, 1),
                                            (1, 2, 8, 16)])
    @pytest.mark.parametrize('dyn_shapes', (True, False))
    def test_scaled_dot_product_atten_fp64(self, ie_device, precision,
                                           ir_version, mask, is_causal,
                                           mask_shape, dyn_shapes):
        if PytorchLayerTest.use_torch_export() and not mask and is_causal:
            pytest.xfail(reason="Unsupported case for torch.export")
        dtype = np.float64
        self._test(*self.create_model(mask, is_causal, dtype, mask_shape),
                   ie_device, precision, ir_version, dynamic_shapes=dyn_shapes,
                   kwargs_to_prepare_input={"dtype": dtype})


class TestScaledDotProductAttentionWithGroupQuery(PytorchLayerTest):
    def _prepare_input(self, dtype):
        # with group size equal to 2 = 6 / 3
        return (self.random.randn(1, 7, 6, 8, 4, dtype=dtype),
                self.random.randn(1, 7, 3, 16, 4, dtype=dtype),
                self.random.randn(1, 7, 3, 16, 6, dtype=dtype))

    def create_model(self, mask, is_causal, dtype, mask_shape):
        import torch.nn.functional as F
        import torch

        # Generate mask outside inner class to use test's seeded random
        attn_mask = None if not mask else torch.from_numpy(
            self.random.randint(0, 2, mask_shape, dtype=dtype))

        class aten_sdpa(torch.nn.Module):
            def __init__(self, mask=False, is_causal=False,
                         dtype=np.float32, mask_shape=None) -> None:
                super().__init__()
                self.mask = attn_mask
                self.is_causal = False if (is_causal and mask) else is_causal

            def forward(self, query, key, value):
                # torch export struggles with dynamic scale
                a = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   enable_gqa=True)
                b = F.scaled_dot_product_attention(query, key, value, attn_mask=self.mask, is_causal=self.is_causal,
                                                   scale=torch.tensor(5, dtype=torch.float), enable_gqa=True)
                return a, b

        return (aten_sdpa(mask, is_causal, dtype, mask_shape),
                'aten::scaled_dot_product_attention')

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('mask', (True, False))
    @pytest.mark.parametrize('is_causal', (True, False))
    @pytest.mark.parametrize('mask_shape', [(8, 16),
                                            (1, 16),
                                            (8, 1),
                                            (1, 1),
                                            (1, 8, 1),
                                            (1, 1, 8, 1),
                                            (1, 1, 8, 16)])
    @pytest.mark.parametrize('dyn_shapes', (True, False))
    def test_scaled_dot_product_atten_with_gqa(self, ie_device, precision, ir_version,
                                               mask, is_causal, mask_shape, dyn_shapes):
        dtype = np.float32
        self._test(*self.create_model(mask, is_causal, dtype, mask_shape),
                   ie_device, precision, ir_version, dynamic_shapes=dyn_shapes,
                   kwargs_to_prepare_input={"dtype": dtype})
