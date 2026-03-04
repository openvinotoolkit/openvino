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
            # For equal min/max, inject one element equal to min_val so the
            # middle-bin counting path (range_is_zero) is actually exercised.
            data = np.random.randn(*input_shape).astype(input_dtype)
            data.flat[0] = min_val
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
    @pytest.mark.parametrize("min_max", [(0, 10), (-5, 5), (0, 100), (0, 0), (5, 5)])
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
    """Tests auto-range path: when min=max=0, histc gets range from the input data."""

    def _prepare_input(self, input_shape, input_dtype):
        # Use a non-trivial range so ReduceMin/ReduceMax produce a meaningful bin_width
        return (np.random.uniform(-5, 5, input_shape).astype(input_dtype),)

    def create_model(self):
        class aten_histc_default(torch.nn.Module):
            def forward(self, input):
                # min=0, max=0 is PyTorch's default, which triggers auto-range from data
                return torch.histc(input, bins=100, min=0, max=0)

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


class TestHistcNanInf(PytorchLayerTest):
    """Tests NaN/Inf handling: these out-of-range values should not be counted in any bin."""

    def _prepare_input(self, input_shape, input_dtype, special_value):
        # Use explicit min/max (not auto-range) to avoid ReduceMin/ReduceMax issues with NaN/Inf
        data = np.random.uniform(0, 10, input_shape).astype(input_dtype)
        # Inject NaN or Inf into first element
        data.flat[0] = special_value
        return (data,)

    def create_model(self, special_value):
        class aten_histc_nan_inf(torch.nn.Module):
            def forward(self, input):
                # Use explicit range [0, 10] so NaN/Inf are outside and should be masked
                return torch.histc(input, bins=10, min=0, max=10)

        ref_net = None
        return aten_histc_nan_inf(), ref_net, "aten::histc"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[20], [5, 4]])
    @pytest.mark.parametrize("input_dtype", ["float32"])
    @pytest.mark.parametrize("special_value", [np.nan, np.inf, -np.inf])
    def test_histc_nan_inf(self, input_shape, input_dtype, special_value, ie_device, precision, ir_version):
        self._test(*self.create_model(special_value), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "input_shape": input_shape,
                       "input_dtype": input_dtype,
                       "special_value": special_value
                   })
