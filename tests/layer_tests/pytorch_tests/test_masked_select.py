# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from packaging.version import parse as parse_version
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestMaskedSelect(PytorchLayerTest):
    def _prepare_input(self, mask_select='ones', mask_dtype=bool, input_dtype=float):
        input_shape = [1, 10]
        mask = np.zeros(input_shape).astype(mask_dtype)
        if mask_select == 'ones':
            mask = np.ones(input_shape).astype(mask_dtype)
        if mask_select == 'random':
            idx = np.random.choice(10, 5)
            mask[:, idx] = 1
        return (np.random.randn(1, 10).astype(input_dtype), mask)

    def create_model(self):
        import torch

        class aten_masked_select(torch.nn.Module):
            def __init__(self):
                super(aten_masked_select, self).__init__()

            def forward(self, x, mask):
                return x.masked_select(mask)

        ref_net = None

        return aten_masked_select(), ref_net, "aten::masked_select"

    @pytest.mark.parametrize(
        "mask_select", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.float64, int, np.int32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_masked_select(self, mask_select, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(),
                   ie_device, precision, ir_version,
                   dynamic_shapes=False,
                   trace_model=True,
                   kwargs_to_prepare_input={'mask_select': mask_select, 'mask_dtype': bool, "input_dtype": input_dtype})

    @pytest.mark.skipif(parse_version(torch.__version__) >= parse_version("2.1.0"), reason="pytorch 2.1 and above does not support nonboolean mask")
    @pytest.mark.parametrize(
        "mask_select", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.float64, int, np.int32])
    @pytest.mark.parametrize("mask_dtype", [np.uint8, np.int32, np.float32])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_masked_select_non_bool_mask(self, mask_select, mask_dtype, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(),
                   ie_device, precision, ir_version,
                   dynamic_shapes=False,
                   trace_model=True,
                   kwargs_to_prepare_input={'mask_select': mask_select, 'mask_dtype': mask_dtype, "input_dtype": input_dtype})
        