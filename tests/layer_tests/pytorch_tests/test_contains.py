# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestContains(PytorchLayerTest):
    """
    Tests for aten::__contains__ operation (tensor membership check).
    
    Implements 'item in tensor' by comparing item with all elements
    and reducing with logical OR.
    """

    def _prepare_input(self, dtype):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        return (x,)

    def create_model(self, item):
        class aten_contains(torch.nn.Module):
            def __init__(self, item):
                super().__init__()
                self.item = item

            def forward(self, x):
                return self.item in x

        return aten_contains(item), None, "aten::__contains__"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype,item", [
        ("int64", 1),      # present
        ("int64", 6),      # present
        ("int64", 7),      # not present
        ("float32", 3.0),  # present
        ("float32", -1.0), # not present
    ])
    def test_contains(self, dtype, item, ie_device, precision, ir_version):
        self._test(
            *self.create_model(item),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype},
        )


class TestContains1D(PytorchLayerTest):
    """Tests for aten::__contains__ with 1D tensors."""

    def _prepare_input(self, dtype):
        x = np.array([10, 20, 30, 40, 50], dtype=dtype)
        return (x,)

    def create_model(self, item):
        class aten_contains_1d(torch.nn.Module):
            def __init__(self, item):
                super().__init__()
                self.item = item

            def forward(self, x):
                return self.item in x

        return aten_contains_1d(item), None, "aten::__contains__"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype,item", [
        ("int64", 30),     # present
        ("int64", 100),    # not present
        ("float64", 20.0), # present
    ])
    def test_contains_1d(self, dtype, item, ie_device, precision, ir_version):
        self._test(
            *self.create_model(item),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"dtype": dtype},
        )
