# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestBincount(PytorchLayerTest):
    def _prepare_input(self, with_weights=False, n=10, max_val=5):
        data = np.random.randint(0, max_val, size=(n,)).astype(np.int32)
        if not with_weights:
            return (data,)
        weights = np.random.randn(n).astype(np.float32)
        return (data, weights)

    def create_model(self, minlength=0, with_weights=False):
        import torch

        class aten_bincount_no_weights(torch.nn.Module):
            def __init__(self, minlength):
                super().__init__()
                self.minlength = minlength

            def forward(self, x):
                return torch.bincount(x, minlength=self.minlength)

        class aten_bincount_with_weights(torch.nn.Module):
            def __init__(self, minlength):
                super().__init__()
                self.minlength = minlength

            def forward(self, x, w):
                return torch.bincount(x, weights=w, minlength=self.minlength)

        if with_weights:
            model_cls = aten_bincount_with_weights
        else:
            model_cls = aten_bincount_no_weights

        ref_net = None
        return model_cls(minlength), ref_net, "aten::bincount"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("minlength", [0, 3, 10])
    def test_bincount_unweighted(self, minlength, ie_device, precision, ir_version):
        self._test(*self.create_model(minlength=minlength, with_weights=False),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"with_weights": False, "n": 15, "max_val": 8})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("minlength", [0, 5])
    def test_bincount_weighted(self, minlength, ie_device, precision, ir_version):
        self._test(*self.create_model(minlength=minlength, with_weights=True),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"with_weights": True, "n": 12, "max_val": 6})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bincount_empty(self, ie_device, precision, ir_version):
        self._test(*self.create_model(minlength=5, with_weights=False),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"with_weights": False, "n": 0, "max_val": 1})
