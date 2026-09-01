# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestHistc(PytorchLayerTest):

    def _prepare_input(self, input_data, input_dtype):
        return (np.array(input_data, dtype=input_dtype),)

    def create_model(self, bins, min_val, max_val):
        class aten_histc(torch.nn.Module):
            def __init__(self, bins, min_val, max_val):
                super().__init__()
                self.bins = bins
                self.min_val = min_val
                self.max_val = max_val

            def forward(self, x):
                return torch.histc(x, bins=self.bins, min=self.min_val, max=self.max_val)

        return aten_histc(bins, min_val, max_val), "aten::histc"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "input_data,bins,min_val,max_val,input_dtype",
        [
            # basic histogram, explicit bins/min/max
            ([1.0, 2.0, 1.0], 4, 0.0, 3.0, np.float32),
            # value exactly equal to min / max; last bin includes max
            ([0.0, 1.0, 2.0, 3.0], 3, 0.0, 3.0, np.float32),
            ([3.0], 3, 0.0, 3.0, np.float32),
            ([0.0], 3, 0.0, 3.0, np.float32),
            # values below min / above max are excluded
            ([-1.0, 0.0, 1.0, 2.0], 2, 0.0, 2.0, np.float32),
            ([0.0, 1.0, 2.0, 3.0], 2, 0.0, 2.0, np.float32),
            # multiple bin counts
            ([1.0, 2.0, 3.0], 1, 0.0, 3.0, np.float32),
            ([1.0, 2.0, 3.0, 4.0], 10, 1.0, 4.0, np.float32),
            # negative and mixed-sign values
            ([-3.0, -1.0, 0.0, 2.0], 5, -3.0, 2.0, np.float32),
            ([-2.0, -0.5, 0.0, 1.5, 4.0], 4, -2.0, 2.0, np.float32),
            # automatic range: min == max == 0 infers from data
            ([1.0, 2.0, 3.0, 4.0], 4, 0.0, 0.0, np.float32),
            # automatic range on a constant tensor expands by ±1
            ([5.0, 5.0, 5.0], 4, 0.0, 0.0, np.float32),
            # min == max != 0 also infers from data (PyTorch histc)
            ([1.0, 2.0, 3.0], 4, 5.0, 5.0, np.float32),
            # 2D is flattened
            ([[1.0, 2.0], [3.0, 4.0]], 4, 1.0, 4.0, np.float32),
            # float64
            ([1.0, 2.0, 3.0], 3, 1.0, 3.0, np.float64),
            # NaN / Inf are excluded
            ([np.nan, 2.0, 3.0], 2, 1.0, 4.0, np.float32),
            ([np.inf, 2.0, 3.0], 2, 1.0, 4.0, np.float32),
            ([-np.inf, 2.0, 3.0], 2, 1.0, 4.0, np.float32),
        ],
    )
    def test_histc(self, input_data, bins, min_val, max_val, input_dtype, ie_device, precision, ir_version):
        self._test(
            *self.create_model(bins, min_val, max_val),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            dynamic_shapes=False,
            kwargs_to_prepare_input={"input_data": input_data, "input_dtype": input_dtype},
        )


class TestHistcDefaultBins(PytorchLayerTest):
    """Default bins=100 with min=max=0 (auto range)."""

    def _prepare_input(self, input_data):
        return (np.array(input_data, dtype=np.float32),)

    def create_model(self):
        class aten_histc_default(torch.nn.Module):
            def forward(self, x):
                return torch.histc(x)

        return aten_histc_default(), "aten::histc"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_default_bins(self, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            dynamic_shapes=False,
            kwargs_to_prepare_input={"input_data": [0.0, 50.0, 100.0]},
        )
