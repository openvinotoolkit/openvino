# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestContains(PytorchLayerTest):
    """
    Tests for aten::__contains__ operation (tensor membership check).

    TorchScript only supports __contains__ on list types (int[], float[], str[]),
    not on Tensors directly. We use trace_model=True and kind=None since the
    op won't appear in the inlined graph; the test verifies E2E correctness
    by comparing PyTorch traced output against OpenVINO inference.
    """

    def _prepare_input(self, dtype):
        x = np.array([[1, 2, 3], [4, 5, 6]], dtype=dtype)
        return (x,)

    def create_model(self, item):
        class aten_contains(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def forward(self, x):
                if self.val in x:
                    return x + 1
                return x - 1

        return aten_contains(item), None, None

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype,item", [
        ("int64", 1),
        ("int64", 6),
        ("int64", 7),
        ("float32", 3.0),
        ("float32", -1.0),
    ])
    def test_contains(self, dtype, item, ie_device, precision, ir_version):
        self._test(
            *self.create_model(item),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            kwargs_to_prepare_input={"dtype": dtype},
        )


class TestContains1D(PytorchLayerTest):
    """Tests for aten::__contains__ with 1D tensors."""

    def _prepare_input(self, dtype):
        x = np.array([10, 20, 30, 40, 50], dtype=dtype)
        return (x,)

    def create_model(self, item):
        class aten_contains_1d(torch.nn.Module):
            def __init__(self, val):
                super().__init__()
                self.val = val

            def forward(self, x):
                if self.val in x:
                    return x + 1
                return x - 1

        return aten_contains_1d(item), None, None

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype,item", [
        ("int64", 30),
        ("int64", 100),
        ("float64", 20.0),
    ])
    def test_contains_1d(self, dtype, item, ie_device, precision, ir_version):
        self._test(
            *self.create_model(item),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            kwargs_to_prepare_input={"dtype": dtype},
        )
