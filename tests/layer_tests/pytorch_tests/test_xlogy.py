# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestXlogy(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(self.x, dtype=self.dtype), np.array(self.y, dtype=self.dtype))

    def create_model(self):
        class aten_xlogy(torch.nn.Module):
            def forward(self, x, y):
                return torch.xlogy(x, y)

        return aten_xlogy(), "aten::xlogy"

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    @pytest.mark.parametrize(("x", "y"), [
        ([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]),
        ([0.0], [0.0]),
        ([0.0, 1.0, 2.0], [1.0, 2.0, 4.0]),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_xlogy(self, dtype, x, y, ie_device, precision, ir_version):
        self.dtype = dtype
        self.x = x
        self.y = y
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.xlogy")

    @pytest.mark.parametrize(("x", "y"), [
        # zeros in x with finite, inf, and NaN y; negative y yields NaN when x != 0
        ([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, np.nan],
         [1.0, np.inf, np.nan, -1.0, 2.0, 0.0, 3.0, 2.0]),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_xlogy_edge_values(self, x, y, ie_device, precision, ir_version):
        if precision == "FP16":
            pytest.skip("Values exceed the f16 range")
        self.dtype = np.float32
        self.x = x
        self.y = y
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.xlogy")


class TestXlogyTypes(PytorchLayerTest):
    def _prepare_input(self):
        lhs = np.abs(self.random.randn(*self.lhs_shape, dtype=self.lhs_type))
        rhs = np.abs(self.random.randn(*self.rhs_shape, dtype=self.rhs_type)) + 1
        return (lhs, rhs)

    def create_model(self, lhs_type, rhs_type):
        class aten_xlogy(torch.nn.Module):
            def __init__(self, lhs_type, rhs_type):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type

            def forward(self, lhs, rhs):
                return torch.xlogy(lhs.to(self.lhs_type), rhs.to(self.rhs_type))

        return aten_xlogy(lhs_type, rhs_type), "aten::xlogy"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.float32, torch.float32],
                              [torch.float32, torch.float64],
                              [torch.float64, torch.float32],
                              [torch.float64, torch.float64],
                              [torch.int32, torch.int32],
                              [torch.int64, torch.int64],
                              [torch.int32, torch.float32],
                              [torch.float32, torch.int32],
                              [torch.int64, torch.float64]])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], [1, 3]),
                                                          ([3, 2, 3], [2, 3])])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_xlogy_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, rhs_type),
                   ie_device, precision, ir_version, freeze_model=False, trace_model=True, fx_kind="aten.xlogy")


class TestXlogyIntInputs(PytorchLayerTest):
    # Integer inputs reach aten::xlogy directly, with no .to(...) cast in the model,
    # so the scripted conversion must promote them to floating point inside the graph.
    def _prepare_input(self):
        return (np.array([0, 1, 2, 3], dtype=self.dtype),
                np.array([0, 2, 3, 4], dtype=self.dtype))

    def create_model(self):
        class aten_xlogy(torch.nn.Module):
            def forward(self, x, y):
                return torch.xlogy(x, y)

        return aten_xlogy(), "aten::xlogy"

    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_xlogy_int_inputs(self, dtype, ie_device, precision, ir_version):
        self.dtype = dtype
        # trace_model=False scripts the model, so parameter types stay dynamic until
        # the test assigns the integer input types after conversion.
        self._test(*self.create_model(), ie_device, precision, ir_version, trace_model=False)


class TestXlogyIntScalar(PytorchLayerTest):
    # An integer python scalar must keep its value: 256 does not fit in uint8,
    # so promoting it to the tensor type before the float cast would give log(0).
    def _prepare_input(self):
        return (np.array([0, 1, 2, 3], dtype=self.dtype),)

    def create_model(self, scalar_first):
        class aten_xlogy_scalar_other(torch.nn.Module):
            def forward(self, x):
                return torch.xlogy(x, 256)

        class aten_xlogy_scalar_self(torch.nn.Module):
            def forward(self, x):
                return torch.xlogy(256, x + 1)

        model = aten_xlogy_scalar_self() if scalar_first else aten_xlogy_scalar_other()
        return model, "aten::xlogy"

    @pytest.mark.parametrize("dtype", [np.uint8, np.int32])
    @pytest.mark.parametrize("scalar_first", [False, True])
    @pytest.mark.parametrize("trace_model", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_xlogy_int_scalar(self, dtype, scalar_first, trace_model, ie_device, precision, ir_version):
        self.dtype = dtype
        self._test(*self.create_model(scalar_first), ie_device, precision, ir_version, trace_model=trace_model)


class TestXlogyBroadcast(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([[1.0, 0.0, 3.0]], dtype=np.float32),
                np.array([[2.0], [4.0]], dtype=np.float32))

    def create_model(self):
        class aten_xlogy(torch.nn.Module):
            def forward(self, x, y):
                return torch.xlogy(x, y)

        return aten_xlogy(), "aten::xlogy"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_xlogy_broadcast(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.xlogy")


class TestXlogyScalar(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([0.0, 1.0, 2.0, np.nan], dtype=np.float32),)

    def create_model(self, scalar_self):
        class aten_xlogy_scalar(torch.nn.Module):
            def __init__(self, scalar_self):
                super().__init__()
                if scalar_self:
                    self.forward = self.forward_scalar_self

            def forward(self, x):
                return torch.xlogy(x, 3.0)

            def forward_scalar_self(self, y):
                return torch.xlogy(2.0, y)

        return aten_xlogy_scalar(scalar_self), "aten::xlogy"

    @pytest.mark.parametrize("scalar_self", [False, True])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_xlogy_scalar(self, scalar_self, ie_device, precision, ir_version):
        self._test(*self.create_model(scalar_self),
                   ie_device, precision, ir_version, fx_kind="aten.xlogy")


class TestXlogyInplace(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

    def create_model(self):
        class aten_xlogy_(torch.nn.Module):
            def forward(self, x, y):
                return x.xlogy_(y)

        return aten_xlogy_(), "aten::xlogy_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_xlogy_inplace(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestXlogyInplaceMixedTypes(PytorchLayerTest):
    # x.xlogy_(y) with float32 x and float64 y computes in float64 but writes back float32,
    # so the following matmul with a float32 weight must still convert.
    def _prepare_input(self):
        return (np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))

    def create_model(self):
        class aten_xlogy_mixed(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.tensor([[1.0, 0.5], [0.25, 2.0]], dtype=torch.float32)

            def forward(self, x, y):
                x.xlogy_(y)
                return x @ self.w, x

        return aten_xlogy_mixed(), "aten::xlogy_"

    @pytest.mark.parametrize("trace_model", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_xlogy_inplace_mixed_types(self, trace_model, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, trace_model=trace_model)


class TestXlogyOut(PytorchLayerTest):
    def _prepare_input(self):
        x = np.array([0.0, 1.0, 2.0], dtype=self.in_dtype)
        y = np.array([1.0, 2.0, 3.0], dtype=self.in_dtype)
        return (x, y, np.zeros(x.shape, dtype=self.out_dtype))

    def create_model(self):
        class aten_xlogy_out(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.out_op = PytorchLayerTest.out_variant(torch.xlogy, torch.ops.aten.xlogy.OutTensor)

            def forward(self, x, y, out):
                return self.out_op(x, y, out=out), out

        return aten_xlogy_out(), "aten::xlogy"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize(("in_dtype", "out_dtype"), [
        (np.float32, np.float32),
        # out keeps its own dtype: float64 operands are rounded into a float32 destination
        (np.float64, np.float32),
    ])
    def test_xlogy_out(self, in_dtype, out_dtype, ie_device, precision, ir_version):
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.xlogy")


class TestXlogySpecial(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32),
                np.array([1.0, np.nan, 2.0, 3.0], dtype=np.float32))

    def create_model(self):
        class aten_special_xlogy(torch.nn.Module):
            def forward(self, x, y):
                return torch.special.xlogy(x, y)

        # TorchScript lowers torch.special.xlogy to aten::xlogy; torch.export keeps
        # aten.special_xlogy.default.
        kind = "aten::special_xlogy" if PytorchLayerTest.use_torch_export() else "aten::xlogy"
        return aten_special_xlogy(), kind

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_xlogy_special(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version, fx_kind="aten.special_xlogy")
