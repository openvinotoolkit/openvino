# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
from packaging.version import parse as parse_version
import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestMaskedFill(PytorchLayerTest):
    def _prepare_input(self, mask_fill='ones', mask_dtype=bool, input_dtype=float):
        input_shape = [1, 10]
        mask = np.zeros(input_shape).astype(mask_dtype)
        if mask_fill == 'ones':
            mask = np.ones(input_shape).astype(mask_dtype)
        if mask_fill == 'random':
            idx = np.random.choice(10, 5)
            mask[:, idx] = 1

        return (np.random.randn(1, 10).astype(input_dtype), mask)

    def create_model(self, value, inplace):
        import torch

        class aten_masked_fill(torch.nn.Module):
            def __init__(self, value):
                super(aten_masked_fill, self).__init__()
                self.value = value

            def forward(self, x, mask):
                return x.masked_fill(mask, self.value)

        class aten_masked_fill_(torch.nn.Module):
            def __init__(self, value):
                super(aten_masked_fill_, self).__init__()
                self.value = value

            def forward(self, x, mask):
                return x.masked_fill_(mask, self.value)

        ref_net = None

        if not inplace:
            return aten_masked_fill(value), ref_net, "aten::masked_fill"
        return aten_masked_fill_(value), ref_net, "aten::masked_fill_"

    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2])
    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.float64, int, np.int32])
    @pytest.mark.parametrize("mask_dtype", [bool])  # np.float32 incorrectly casted to bool
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_masked_fill(self, value, mask_fill, mask_dtype, input_dtype, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(value, inplace),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'mask_fill': mask_fill, 'mask_dtype': mask_dtype, "input_dtype": input_dtype})

    @pytest.mark.skipif(parse_version(torch.__version__) >= parse_version("2.1.0"), reason="pytorch 2.1 and above does not support nonboolean mask")
    @pytest.mark.parametrize("value", [0.0, 1.0, -1.0, 2])
    @pytest.mark.parametrize(
        "mask_fill", ['zeros', 'ones', 'random'])
    @pytest.mark.parametrize("input_dtype", [np.float32, np.float64, int, np.int32])
    @pytest.mark.parametrize("mask_dtype", [np.uint8, np.int32])  # np.float32 incorrectly casted to bool
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_masked_fill_non_bool_mask(self, value, mask_fill, mask_dtype, input_dtype, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(value, inplace),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={'mask_fill': mask_fill, 'mask_dtype': mask_dtype, "input_dtype": input_dtype})
