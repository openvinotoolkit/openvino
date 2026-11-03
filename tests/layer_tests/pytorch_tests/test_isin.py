# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestIsin(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randint(-3, 5, (2, 3), dtype="int64"),
                self.random.randint(-3, 5, (0 if self.empty else 4,), dtype="int64"))

    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("invert", [False, True])
    @pytest.mark.parametrize("empty", [False, True])
    def test_isin(self, ie_device, precision, ir_version, invert, empty):
        class Isin(torch.nn.Module):
            def forward(self, elements, test_elements):
                return torch.isin(elements, test_elements, invert=invert)

        self.empty = empty
        self._test(Isin(), "aten::isin", ie_device, precision, ir_version, trace_model=True)


class TestIsinScalar(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([[1, 2, 3], [2, -1, 0]], dtype=np.float32),)

    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("scalar_elements", [False, True])
    @pytest.mark.parametrize("invert", [False, True])
    def test_isin_scalar(self, ie_device, precision, ir_version, scalar_elements, invert):
        class Isin(torch.nn.Module):
            def forward(self, value):
                if scalar_elements:
                    return torch.isin(2, value, invert=invert)
                return torch.isin(value, 2, invert=invert)

        self._test(Isin(), "aten::isin", ie_device, precision, ir_version, trace_model=True)
