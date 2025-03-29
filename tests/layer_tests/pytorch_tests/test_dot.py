# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest

class TestDot(PytorchLayerTest):
    def _prepare_input(self, inputs, dtype, out=False):
        import numpy as np
        x = np.array(inputs[0]).astype(dtype)
        y = np.array(inputs[1]).astype(dtype)
        if not out:
            return (x, y)
        return (x, y, np.array(0).astype(dtype))

    def create_model(self, mode, dtype):
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
        }

        class aten_dot(torch.nn.Module):
            def __init__(self, mode, dtype):
                super().__init__()
                self.dtype = dtype
                if mode =="out":
                    self.forward = self.forward_out
                else:
                    self.forward = self.forward_default

            def forward_default(self, tensor1, tensor2):
                return torch.dot(tensor1.to(self.dtype), tensor2.to(self.dtype))

            def forward_out(self, tensor1, tensor2, y):
                return torch.dot(tensor1.to(self.dtype), tensor2.to(self.dtype), out=y), y

        dtype = dtype_map.get(dtype)

        ref_net = None

        return aten_dot(mode, dtype), ref_net, "aten::dot"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("mode, dtype", [
        ("", "float32"), ("", "float64"), ("", "int32"), ("", "int64"), ("", "int8"),
        ("out", "float32"), ("out", "float64"), ("out", "int32"), ("out", "int64"), ("out", "int8")])
    @pytest.mark.parametrize(
        "inputs", [([0, 1, 2, 3, 4], [5, 6, 7, 8, 9]), ([1, 2, 3], [4, 5, 6]), ([1, 1, 1], [1, 1, 1])]
    )
    def test_dot(self, mode, dtype, inputs, ie_device, precision, ir_version):
        self._test(
            *self.create_model(mode, dtype),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"inputs": inputs, "dtype": dtype, "out": mode == "out"}
        )
