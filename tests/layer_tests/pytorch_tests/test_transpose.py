# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestTranspose(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 3, 4, 5).astype(np.float32),)

    def create_model(self, dim0, dim1, op_type):
        class swapaxes(torch.nn.Module):
            def __init__(self, dim0, dim1):
                super().__init__()
                self.dim0 = dim0
                self.dim1 = dim1

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.swapaxes(x, self.dim0, self.dim1)

        class aten_transpose(torch.nn.Module):
            def __init__(self, dim0, dim1, op_type):
                super(aten_transpose, self).__init__()
                self.dim0 = dim0
                self.dim1 = dim1
                op_types = {"transpose": self.forward_transpose, "swapaxes": self.forward_swapaxes}
                self.swapaxes = swapaxes(dim0, dim1)
                self.forward = op_types.get(op_type)

            def forward_transpose(self, x):
                return torch.transpose(x, self.dim0, self.dim1)

            def forward_swapaxes(self, x: torch.Tensor) -> torch.Tensor:
                # To reproduce aten::swapaxes in graph, swapaxes need to be in separate graph and tracing need to be used.
                return self.swapaxes(x)

        ref_net = None

        return aten_transpose(dim0, dim1, op_type), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize("dim0", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("dim1", [0, 1, 2, 3, -1, -2, -3, -4])
    @pytest.mark.parametrize("op_type", ["transpose", "swapaxes"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_transpose(self, dim0, dim1, op_type, ie_device, precision, ir_version):
        self._test(*self.create_model(dim0, dim1, op_type), ie_device, precision, ir_version, trace_model=True)


class TestMoveDim(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 3, 4, 5).astype(np.float32),)

    def create_model(self, dim0, dim1):
        class aten_move_dim(torch.nn.Module):
            def __init__(self, dim0, dim1):
                super(aten_move_dim, self).__init__()
                self.dim0 = dim0
                self.dim1 = dim1

            def forward(self, x):
                return torch.movedim(x, self.dim0, self.dim1)

        ref_net = None

        return aten_move_dim(dim0, dim1), ref_net, f"aten::movedim"

    @pytest.mark.parametrize(("dim0", "dim1"), [[0, 1], [-1, 0], [2, -2], [3, 1], [3, 3], [[1, 2], [3, 0]], [[-4, 1], [1, -1]], [[1, 3, 2], [0, 1, 2 ]]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_move_dim(self, dim0, dim1, ie_device, precision, ir_version):
        self._test(*self.create_model(dim0, dim1), ie_device, precision, ir_version, trace_model=True)

class TestTSmall(PytorchLayerTest):
    def _prepare_input(self, num_dims=2, input_dtype="float32"):
        shape = (2, 3)
        if num_dims == 0:
            return (np.array(num_dims).astype(input_dtype),)
        return (np.random.randn(*shape[:num_dims]).astype(input_dtype),)

    def create_model(self, mode):
        class aten_transpose(torch.nn.Module):
            def __init__(self, mode):
                super(aten_transpose, self).__init__()
                if mode == "inplace":
                    self.forward = self.forward_inplace
                elif mode == "numpy":
                    self.forward = self.forward_numpy_t

            def forward(self, x):
                return x.t(), x

            def forward_inplace(self, x):
                return x.t_(), x

            def forward_numpy_t(self, x):
                return x.T, x

        ref_net = None

        return aten_transpose(mode), ref_net, "aten::t_" if mode == "inplace" else ("aten::numpy_T" if mode == "numpy" else "aten::t")

    @pytest.mark.parametrize("num_dims", [0, 1, 2])
    @pytest.mark.parametrize("input_dtype", ["float32", "int32"])
    @pytest.mark.parametrize("mode", [None, "inplace", "numpy"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_t_small(self, num_dims, input_dtype, mode, ie_device, precision, ir_version):
        self._test(
            *self.create_model(mode),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"num_dims": num_dims, "input_dtype": input_dtype},
            use_convert_model=True,
        )
