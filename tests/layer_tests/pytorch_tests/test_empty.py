# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestEmptyNumeric(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 3),)

    def create_model(self, dtype):

        class aten_empty(torch.nn.Module):

            def __init__(self, dtype) -> None:
                dtype_map = {
                    "float32": torch.float32,
                    "float64": torch.float64,
                    "int64": torch.int64,
                    "int32": torch.int32,
                    "uint8": torch.uint8,
                    "int8": torch.int8
                }
                super().__init__()
                self.dtype = dtype_map[dtype]
                self.zero = torch.tensor([0], dtype=dtype_map[dtype])

            def forward(self, input_tensor):
                size = input_tensor.shape
                empty = torch.empty(size, dtype=self.dtype)
                # We don't want to compare values, just shape and type,
                # so we multiply the tensor by zero.
                return empty*self.zero

        ref_net = None

        return aten_empty(dtype), ref_net, "aten::empty"

    @pytest.mark.parametrize('dtype', ("float32", "float64", "int64", "int32", "uint8", "int8"))
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_empty(self, ie_device, precision, ir_version, dtype):
        self._test(*self.create_model(dtype), ie_device, precision, ir_version)

class TestEmptyBoolean(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(3, 4, 3),)

    def create_model(self):

        class aten_empty(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()
                self.false = torch.tensor([False])

            def forward(self, input_tensor):
                size = input_tensor.shape
                empty = torch.empty(size, dtype=torch.bool)
                # We don't want to compare values, just shape and type,
                # so we do "and" operation with False.
                return empty & self.false

        ref_net = None

        return aten_empty(), ref_net, "aten::empty"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_empty_bool(self, ie_device, precision, ir_version, ):
        self._test(*self.create_model(), ie_device, precision, ir_version)
