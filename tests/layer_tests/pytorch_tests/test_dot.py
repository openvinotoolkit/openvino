# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class TestDot(PytorchLayerTest):
    def _prepare_input(self, inputs, dtype):
        return ((np.array(inputs[0]).astype(dtype)),
                (np.array(inputs[1]).astype(dtype)))

    def create_model(self, dtype):

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
        }

        class aten_dot(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, tensor1, tensor2):
                return torch.dot(tensor1.to(self.dtype), tensor2.to(self.dtype))

        dtype = dtype_map.get(dtype)
        
        ref_net = None

        return aten_dot(dtype), ref_net, "aten::dot"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize(
        "inputs", [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), ([1, 2, 3], [4, 5, 6]), ([1, 1, 1], [1, 1, 1])]
    )
    def test_dot(self, dtype, inputs, ie_device, precision, ir_version):
        self.dtype = dtype
        self.inputs = inputs
        self._test(
            *self.create_model(dtype),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"inputs": inputs, "dtype": dtype}
        )
