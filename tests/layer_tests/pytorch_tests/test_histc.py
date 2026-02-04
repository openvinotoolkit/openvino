# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestHistc(PytorchLayerTest):

    def _prepare_input(self, input_shape, input_dtype, min_val, max_val):
        if min_val < max_val:
            data = np.random.uniform(min_val, max_val, input_shape).astype(input_dtype)
        else:
            data = np.random.randn(*input_shape).astype(input_dtype)
        return (data,)

    def create_model(self, bins, min_val, max_val):
        class aten_histc(torch.nn.Module):
            def __init__(self, bins, min_val, max_val):
                super().__init__()
                self.bins = bins
                self.min_val = min_val
                self.max_val = max_val

            def forward(self, input):
                return torch.histc(input, bins=self.bins, min=self.min_val, max=self.max_val)

        ref_net = None
        return aten_histc(bins, min_val, max_val), ref_net, "aten::histc"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[10], [5, 5], [2, 3, 4]])
    @pytest.mark.parametrize("input_dtype", ["float32"])
    @pytest.mark.parametrize("bins", [10, 50, 100])
    @pytest.mark.parametrize("min_max", [(0, 10), (-5, 5), (0, 100)])
    def test_histc(self, input_shape, input_dtype, bins, min_max, ie_device, precision, ir_version):
        min_val, max_val = min_max
        self._test(*self.create_model(bins, min_val, max_val), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "input_shape": input_shape,
                       "input_dtype": input_dtype,
                       "min_val": min_val,
                       "max_val": max_val
                   })


class TestHistcDefaultParams(PytorchLayerTest):

    def _prepare_input(self, input_shape, input_dtype):
        return (np.random.uniform(0, 100, input_shape).astype(input_dtype),)

    def create_model(self):
        class aten_histc_default(torch.nn.Module):
            def forward(self, input):
                # Use default parameters: bins=100, min=0, max=0
                return torch.histc(input, bins=100, min=0, max=100)

        ref_net = None
        return aten_histc_default(), ref_net, "aten::histc"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[100], [10, 10]])
    @pytest.mark.parametrize("input_dtype", ["float32"])
    def test_histc_default(self, input_shape, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "input_shape": input_shape,
                       "input_dtype": input_dtype
                   })
