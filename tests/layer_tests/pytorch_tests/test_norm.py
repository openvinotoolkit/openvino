# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch
from packaging import version

from pytorch_layer_test_class import PytorchLayerTest


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

    def create_model_tensor_norm(self, p, dim, keepdim):
        class aten_norm(torch.nn.Module):

            def __init__(self, p, dim, keepdim) -> None:
                super().__init__()
                self.p = p
                self.dim = dim
                self.keepdim = keepdim
                if self.keepdim is None or self.dim is None:
                    self.forward = self.forward2
                else:
                    self.forward = self.forward4

            def forward4(self, input_data):
                return input_data.norm(self.p, self.dim, self.keepdim)

            def forward2(self, input_data):
                return input_data.norm(self.p)

        ref_net = None

        return aten_norm(p, dim, keepdim), ref_net, "aten::norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('p', [-2, -1, 0, 1, 2, 2.5, float('inf'), float('-inf')])
    @pytest.mark.parametrize('dim', [[0], [0, 1], [0, 1, 2]])
    @pytest.mark.parametrize('keepdim', [True, False])
    def test_norm(self, ie_device, precision, ir_version, p, dim, keepdim):
        self._test(*self.create_model(p, dim, keepdim),
                   ie_device, precision, ir_version)

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('p', [-2, -1, 0, 1, 2, 2.5, float('inf'), float('-inf')])
    @pytest.mark.parametrize('dim', [None, [0], [0, 1], [0, 1, 2]])
    @pytest.mark.parametrize('keepdim', [None, True, False])
    def test_norm_tensor(self, ie_device, precision, ir_version, p, dim, keepdim):
        self._test(*self.create_model_tensor_norm(p, dim, keepdim),
                   ie_device, precision, ir_version)

class TestWeightNorm(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(1, 60, 20).astype(np.float32),)

    def create_model(self):
        from torch import nn
        from torch.nn.utils import weight_norm

        return weight_norm(nn.Linear(20, 40), name='weight'), None, "aten::_weight_norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_weight_norm(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, trace_model=True, freeze_model=False)


class TestFrobeniusNorm(PytorchLayerTest):
    def _prepare_input(self, out=False, dtype="float32"):
        x = np.random.randn(10, 12, 14).astype(dtype)
        if not out:
            return (x,)
        y = np.zeros_like(x)
        return (x, y)

    def create_model(self, dim, keepdim, out):
        class aten_frobenius_norm(torch.nn.Module):

            def __init__(self, dim, keepdim, out) -> None:
                super().__init__()
                self.dim = dim
                self.keepdim = keepdim
                if out:
                    self.forward = self.forward_out

            def forward(self, input_data):
                return torch._VF.frobenius_norm(input_data, self.dim, self.keepdim)

            def forward_out(self, input_data, out):
                return torch._VF.frobenius_norm(input_data, self.dim, self.keepdim, out=out), out

        ref_net = None

        return aten_frobenius_norm(dim, keepdim, out), ref_net, "aten::frobenius_norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize('dim', [(1, ), (0, ), (-1, ), (0, 1), (1, 0)])
    @pytest.mark.parametrize('keepdim', [True, False])
    @pytest.mark.parametrize("out", [False, True])
    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_frobenius_norm(self, ie_device, precision, ir_version, dim, keepdim, out, dtype):
        self._test(*self.create_model(dim, keepdim, out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"out": out, "dtype": dtype}
                   )


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
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122715')
    def test_linalg_matrix_norm(self, p, dim, keepdim, dtype, out, prim_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(p, dim, keepdim, dtype, out, prim_dtype),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"out": out or prim_dtype, "out_dtype": dtype if prim_dtype else None})


class TestLinalgNorm(PytorchLayerTest):

    def _prepare_input(self, out=False, out_dtype=None, input_shape=(3, 3)):
        if not out:
            return (np.random.randn(*input_shape).astype(np.float32),)
        x = np.random.randn(*input_shape).astype(np.float32)
        y = np.random.randn(*input_shape).astype(
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
    @pytest.mark.parametrize("input_shape", [[1, 3], [3, 3], [1, 3, 3]])
    def test_linalg_norm(self, p, dim, keepdim, dtype, out, prim_dtype, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(p, dim, keepdim, dtype, out, prim_dtype),
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={
                       "out": out or prim_dtype,
                       "out_dtype": dtype if prim_dtype else None,
                       "input_shape": input_shape
        })


class TestTrickyNorm(PytorchLayerTest):

    def _prepare_input(self, input_shape=(3, 3)):
        return (np.random.randn(*input_shape).astype(np.float32),)

    def create_model(self):
        class aten_norm(torch.nn.Module):
            def forward(self, x):
                return torch.nn.functional.normalize(x, eps=2)

        return aten_norm(), None, ["aten::linalg_vector_norm", "aten::clamp_min"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("input_shape", [[15, 15, 17]])
    def test_tricky_norm(self, input_shape, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_shape": input_shape}, use_convert_model=True, trace_model=True)


class TestRMSNorm(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(2, 5, 10, 10).astype(np.float32),)

    def create_model(self, normalized_shape, eps, gamma):
        class aten_rms_norm(torch.nn.Module):
            def __init__(self, normalized_shape, eps, gamma) -> None:
                super().__init__()
                self.rms = torch.nn.RMSNorm(normalized_shape,
                                            eps=eps,
                                            elementwise_affine=gamma)

            def forward(self, input_data):
                return self.rms(input_data)

        return aten_rms_norm(normalized_shape, eps, gamma), None, "aten::rms_norm"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.skipif(version.parse(torch.__version__) < version.parse("2.4"),
                        reason="Not supported in PyTorch versions earlier than 2.4.")
    @pytest.mark.parametrize("normalized_shape", [[10,],
                                                  [10, 10],
                                                  [5, 10, 10]])
    @pytest.mark.parametrize('gamma', [True, False])
    @pytest.mark.parametrize('eps', [None, 1e-5])
    def test_rms_norm(self, ie_device, precision, ir_version,
                      normalized_shape, eps, gamma):
        self._test(*self.create_model(normalized_shape, eps, gamma),
                   ie_device, precision, ir_version)
