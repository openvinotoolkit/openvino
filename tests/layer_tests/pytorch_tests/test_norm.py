# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('p', [-2, -1, 0, 1, 2, 2.5, float('inf'), float('-inf')])
@pytest.mark.parametrize('dim', [[0], [0, 1], [0, 1, 2]])
@pytest.mark.parametrize('keepdim', [True, False])
class TestNorm(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(1, 2, 3).astype(np.float32),)

    def create_model(self, p, dim, keepdim):
        class aten_norm(torch.nn.Module):

            def __init__(self, p, dim, keepdim) -> None:
                super().__init__()
                self.p = p
                self.dim = dim
                self.keepdim = keepdim

            def forward(self, input_data):
                return torch._VF.norm(input_data, self.p, self.dim, self.keepdim)

        ref_net = None

        return aten_norm(p, dim, keepdim), ref_net, "aten::norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_norm(self, ie_device, precision, ir_version, p, dim, keepdim):
        self._test(*self.create_model(p, dim, keepdim),
                   ie_device, precision, ir_version)


class TestLinalgVectorNorm(PytorchLayerTest):

    def _prepare_input(self, out=False, out_dtype=None):
        if not out:
            return (np.random.randn(1, 2, 3).astype(np.float32),)
        x = np.random.randn(1, 2, 3).astype(np.float32)
        y = np.random.randn(1, 2, 3).astype(
            out_dtype if out_dtype is not None else np.float32)
        return (x, y)

    def create_model(self, p, dim, keepdim, dtype_str, out, out_as_dtype):
        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64
        }
        dtype = dtypes.get(dtype_str)

        class aten_linalg_vector_norm(torch.nn.Module):

            def __init__(self, p, dim, keepdim, dtype, out, out_as_dtype) -> None:
                super().__init__()
                self.ord = p
                self.dim = dim
                self.keepdim = keepdim
                self.dtype = dtype
                if self.dtype is not None:
                    self.forward = self.forward_dtype
                if out:
                    self.forward = self.forward_out
                if out_as_dtype:
                    self.forward = self.forward_prim_dtype

            def forward(self, x):
                return torch.linalg.vector_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

            def forward_dtype(self, x):
                return torch.linalg.vector_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype=self.dtype
                )

            def forward_prim_dtype(self, x, y):
                return torch.linalg.vector_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype=y.dtype
                )

            def forward_out(self, x, y):
                return y, torch.linalg.vector_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, out=y
                )

        ref_net = None

        return aten_linalg_vector_norm(p, dim, keepdim, dtype, out, out_as_dtype), ref_net, "aten::linalg_vector_norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('p', [-2, -1, 0, 1, 2, 2.5, float('inf'), float('-inf')])
    @pytest.mark.parametrize('dim', [0, [0, 1], None])
    @pytest.mark.parametrize('keepdim', [True, False])
    @pytest.mark.parametrize("dtype", ["float32", "float64", None])
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.parametrize("prim_dtype", [True, False])
    def test_linalg_vector_norm(self, p, dim, keepdim, dtype, out, prim_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(p, dim, keepdim, dtype, out, prim_dtype),
                   ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"out": out or prim_dtype, "out_dtype": dtype if prim_dtype else None})


class TestLinalgMatrixNorm(PytorchLayerTest):

    def _prepare_input(self, out=False, out_dtype=None):
        if not out:
            return (np.random.randn(3, 3).astype(np.float32),)
        x = np.random.randn(1, 3, 3).astype(np.float32)
        y = np.random.randn(1, 3, 3).astype(
            out_dtype if out_dtype is not None else np.float32)
        return (x, y)

    def create_model(self, p, dim, keepdim, dtype_str, out, out_as_dtype):
        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64
        }
        dtype = dtypes.get(dtype_str)

        class aten_linalg_matrix_norm(torch.nn.Module):

            def __init__(self, p, dim, keepdim, dtype, out, out_as_dtype) -> None:
                super().__init__()
                self.ord = p
                self.dim = dim
                self.keepdim = keepdim
                self.dtype = dtype
                if self.dtype is not None:
                    self.forward = self.forward_dtype
                if out:
                    self.forward = self.forward_out
                if out_as_dtype:
                    self.forward = self.forward_prim_dtype

            def forward(self, x):
                return torch.linalg.matrix_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

            def forward_dtype(self, x):
                return torch.linalg.matrix_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype=self.dtype
                )

            def forward_prim_dtype(self, x, y):
                return torch.linalg.matrix_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype=y.dtype
                )

            def forward_out(self, x, y):
                return y, torch.linalg.matrix_norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, out=y
                )

        ref_net = None

        return aten_linalg_matrix_norm(p, dim, keepdim, dtype, out, out_as_dtype), ref_net, "aten::linalg_matrix_norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('p', [-1, 1, float('inf'), float('-inf'), "fro"])
    @pytest.mark.parametrize('dim', [[0, 1], [-1, -2]])
    @pytest.mark.parametrize('keepdim', [True, False])
    @pytest.mark.parametrize("dtype", ["float32", "float64", None])
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.parametrize("prim_dtype", [True, False])
    def test_linalg_matrix_norm(self, p, dim, keepdim, dtype, out, prim_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(p, dim, keepdim, dtype, out, prim_dtype),
                   ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"out": out or prim_dtype, "out_dtype": dtype if prim_dtype else None})


class TestLinalgNorm(PytorchLayerTest):

    def _prepare_input(self, out=False, out_dtype=None):
        if not out:
            return (np.random.randn(3, 3).astype(np.float32),)
        x = np.random.randn(3, 3).astype(np.float32)
        y = np.random.randn(3, 3).astype(
            out_dtype if out_dtype is not None else np.float32)
        return (x, y)

    def create_model(self, p, dim, keepdim, dtype_str, out, out_as_dtype):
        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64
        }
        dtype = dtypes.get(dtype_str)

        class aten_linalg_matrix_norm(torch.nn.Module):

            def __init__(self, p, dim, keepdim, dtype, out, out_as_dtype) -> None:
                super().__init__()
                self.ord = p
                self.dim = dim
                self.keepdim = keepdim
                self.dtype = dtype
                if self.dtype is not None:
                    self.forward = self.forward_dtype
                if out:
                    self.forward = self.forward_out
                if out_as_dtype:
                    self.forward = self.forward_prim_dtype

            def forward(self, x):
                return torch.linalg.norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim
                )

            def forward_dtype(self, x):
                return torch.linalg.norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype=self.dtype
                )

            def forward_prim_dtype(self, x, y):
                return torch.linalg.norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, dtype=y.dtype
                )

            def forward_out(self, x, y):
                return y, torch.linalg.norm(
                    x, ord=self.ord, dim=self.dim, keepdim=self.keepdim, out=y
                )

        ref_net = None

        return aten_linalg_matrix_norm(p, dim, keepdim, dtype, out, out_as_dtype), ref_net, "aten::linalg_norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('p,dim', [
        (-1, [0, 1]), (1, [-1, -2]), (float('inf'), [1, 0]), 
        (float('-inf'), [-2, -1]), (0, 1), (1, -1), 
        (None, None), (2.5, 0), (-1, 1), (2, 0), 
        (float('inf'), 1), (float('-inf'), 1), ("fro", (0, 1))])
    @pytest.mark.parametrize('keepdim', [True, False])
    @pytest.mark.parametrize("dtype", ["float32", "float64", None])
    @pytest.mark.parametrize("out", [True, False])
    @pytest.mark.parametrize("prim_dtype", [True, False])
    def test_linalg_norm(self, p, dim, keepdim, dtype, out, prim_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(p, dim, keepdim, dtype, out, prim_dtype),
                   ie_device, precision, ir_version, 
                   kwargs_to_prepare_input={"out": out or prim_dtype, "out_dtype": dtype if prim_dtype else None})
