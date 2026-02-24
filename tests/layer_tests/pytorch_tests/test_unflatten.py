# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestUnflatten(PytorchLayerTest):
    def _prepare_input(self, dtype):
        return (self.random.uniform(0, 50, (6, 3, 4), dtype=dtype),)

    def create_model(self, dim, shape):
        import torch

        class aten_unflatten(torch.nn.Module):
            def __init__(self, dim, shape):
                super().__init__()
                self.dim = dim
                self.shape = shape

            def forward(self, x):
                return x.unflatten(self.dim, self.shape)


        return aten_unflatten(dim, shape), "aten::unflatten"

    @pytest.mark.parametrize(("dim", "shape"), [(0, [2, 1, 3]),  (1, [1, 3]), (2, (2, -1)), (-1, (2, 2)), (-2, (-1, 1))])
    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_unflatten(self, dim, shape, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dim, shape), ie_device, precision, ir_version, kwargs_to_prepare_input={"dtype": dtype})


class TestUnflattenListSizes(PytorchLayerTest):
    def _prepare_input(self, dtype):
        return (self.random.uniform(0, 50, (6, 2, 4), dtype=dtype),)

    def create_model(self, dim):
        import torch

        class aten_unflatten(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, x):
                dim1, dim2, dim3 = x.shape
                if self.dim == 0:
                    sizes = [dim1, -1]
                elif self.dim == 1:
                    sizes = [dim2 // 2, -1]
                else:
                    sizes = [2, dim3 // 2, -1]
                return x.unflatten(self.dim, sizes)


        return aten_unflatten(dim), "aten::unflatten"

    @pytest.mark.parametrize("dim", [0, 1, 2])
    @pytest.mark.parametrize("dtype", ["float32", "int32"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_unflatten(self, dim, dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(dim), ie_device, precision, ir_version, kwargs_to_prepare_input={"dtype": dtype})
