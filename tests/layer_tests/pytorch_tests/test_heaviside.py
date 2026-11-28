# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestHeaviside(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(self.x, dtype=self.dtype), np.array(self.values, dtype=self.dtype))

    def create_model(self):
        class aten_heaviside(torch.nn.Module):
            def forward(self, x, values):
                return torch.heaviside(x, values)

        return aten_heaviside(), "aten::heaviside"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(("x", "values"), [
        ([-2.0, -1.0, 0.0, 1.0, 2.0], [0.5, 0.25, 0.1, 0.3, 0.9]),
        ([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]),
        ([-5.0, -1.0], [9.0, 8.0]),
        ([np.nan, -1.0, 0.0, 1.0, np.inf, -np.inf],
         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_heaviside_float(self, dtype, x, values, ie_device, precision, ir_version):
        # torch.heaviside uses `x == 0 ? v : (x > 0)`, so a NaN input yields 0.
        if any(isinstance(val, float) and np.isnan(val) for val in x):
            torch_nan = torch.heaviside(
                torch.tensor([float("nan")], dtype=torch.float32),
                torch.tensor([0.5], dtype=torch.float32))
            assert torch_nan.item() == 0.0
            assert not torch.isnan(torch_nan).item()
        self.dtype = dtype
        self.x = x
        self.values = values
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.heaviside")

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    @pytest.mark.parametrize(("x", "values"), [
        ([-2, -1, 0, 1, 2], [7, 7, 7, 7, 7]),
        ([0, 0], [3, 4]),
        ([-5, -1], [9, 8]),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_heaviside_int(self, dtype, x, values, ie_device, precision, ir_version):
        self.dtype = dtype
        self.x = x
        self.values = values
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.heaviside")


class TestHeavisideBroadcast(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(self.x, dtype=np.float32), np.array(self.values, dtype=np.float32))

    def create_model(self):
        class aten_heaviside(torch.nn.Module):
            def forward(self, x, values):
                return torch.heaviside(x, values)

        return aten_heaviside(), "aten::heaviside"

    @pytest.mark.parametrize(("x", "values"), [
        ([-1.0, 0.0, 1.0], 0.5),
        ([-1.0, 0.0, 1.0], [0.5]),
        ([[-1.0, 0.0, 1.0]], [[0.2], [0.8]]),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_heaviside_broadcast(self, x, values, ie_device, precision, ir_version):
        self.x = x
        self.values = values
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.heaviside")


class TestHeavisideInplace(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([-1.0, 0.0, 1.0], dtype=np.float32),
                np.array([0.2, 0.3, 0.4], dtype=np.float32))

    def create_model(self):
        class aten_heaviside_(torch.nn.Module):
            def forward(self, x, values):
                return x.heaviside_(values), x

        return aten_heaviside_(), "aten::heaviside_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_heaviside_inplace(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestHeavisideOut(PytorchLayerTest):
    def _prepare_input(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.2, 0.3, 0.4], dtype=np.float32)
        return (x, values, np.zeros_like(x))

    def create_model(self):
        class aten_heaviside_out(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_op = PytorchLayerTest.out_variant(torch.heaviside, torch.ops.aten.heaviside.out)

            def forward(self, x, values, out):
                return self.out_op(x, values, out=out), out

        return aten_heaviside_out(), "aten::heaviside"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_heaviside_out(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.heaviside")
