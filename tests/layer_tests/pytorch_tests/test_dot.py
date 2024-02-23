# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class TestDot(PytorchLayerTest):
    def _prepare_input(self):
        return [torch.tensor(self.inputs[0], dtype=self.dtype1),
                torch.tensor(self.inputs[1], dtype=self.dtype2)]

    def create_model(self, dtype1, dtype2):

        class aten_dot(torch.nn.Module):
            def __init__(self, dtype1, dtype2):
                super().__init__()
                self.dtype1 = dtype1
                self.dtype2 = dtype2

            def forward(self, tensor1, tensor2):
                return torch.dot(tensor1.to(self.dtype1), tensor2.to(self.dtype2))

        ref_net = None

        return aten_dot(dtype1, dtype2), ref_net, "aten::dot"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(("dtype1", "dtype2"),
                             [[torch.int32, torch.int64],
                              [torch.int32, torch.float32],
                              [torch.int32, torch.float64],
                              [torch.int64, torch.int32],
                              [torch.int64, torch.float32],
                              [torch.int64, torch.float64],
                              [torch.float32, torch.int32],
                              [torch.float32, torch.int64],
                              [torch.float32, torch.float64],
                              [torch.float16, torch.uint8],
                              [torch.uint8, torch.float16],
                              [torch.float16, torch.int32],
                              [torch.int32, torch.float16],
                              [torch.float16, torch.int64],
                              [torch.int64, torch.float16]
                              ])
    @pytest.mark.parametrize(
        "inputs", [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), ([1, 2, 3], [4, 5, 6]), ([1, 1, 1], [1, 1, 1])]
    )
    def test_dot(self, dtype1, dtype2, inputs, ie_device, precision, ir_version):
        self.dtype1 = dtype1
        self.dtype2 = dtype2
        self.inputs = inputs
        self._test(
            *self.create_model(dtype1, dtype2),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"inputs": inputs, "dtype1": dtype1, "dtype2": dtype2}
        )
