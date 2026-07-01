# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestHistc(PytorchLayerTest):
    def _prepare_input(self, n=20, low=-5.0, high=5.0):
        data = np.random.uniform(low, high, size=(n,)).astype(np.float32)
        return (data,)

    def create_model(self, bins=100, min_val=0.0, max_val=0.0):
        import torch

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
    @pytest.mark.parametrize("bins", [10, 50, 100])
    def test_histc_auto_range(self, bins, ie_device, precision, ir_version):
        # min == max == 0 -> range derived automatically from the data.
        self._test(*self.create_model(bins=bins, min_val=0.0, max_val=0.0),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"n": 30, "low": -5.0, "high": 5.0})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("min_val,max_val", [(-2.0, 2.0), (0.0, 10.0)])
    def test_histc_explicit_range(self, min_val, max_val, ie_device, precision, ir_version):
        self._test(*self.create_model(bins=8, min_val=min_val, max_val=max_val),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"n": 25, "low": -5.0, "high": 5.0})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_out_of_range_values_excluded(self, ie_device, precision, ir_version):
        # Values outside [min, max] must not be counted in any bin.
        self._test(*self.create_model(bins=4, min_val=0.0, max_val=4.0),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"n": 40, "low": -20.0, "high": 20.0})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_histc_empty(self, ie_device, precision, ir_version):
        # An empty "data" input must still produce a fully-populated, all-zero
        # histogram of size "bins" (output shape does not depend on input size).
        self._test(*self.create_model(bins=5, min_val=0.0, max_val=10.0),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"n": 0, "low": -5.0, "high": 5.0})
