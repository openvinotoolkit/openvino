# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestMul(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_array.astype(self.input_type), self.other_array.astype(self.other_type))

    def create_model(self):
        class aten_mul(torch.nn.Module):
            def __init__(self):
                super(aten_mul, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.mul(input_tensor, other_tensor)

        ref_net = None

        return aten_mul(), ref_net, "aten::mul"

    @pytest.mark.parametrize(("input_array", "other_array"), [
        [np.array([0.2015, -0.4255, 2.6087]), np.array(100)],
        [np.array([[1.1207], [-0.3137], [0.0700], [0.8378]]),
         np.array([[0.5146, 0.1216, -0.5244, 2.2382]])],
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_mul_pt_spec(self, input_array, other_array, ie_device, precision, ir_version):
        self.input_array = input_array
        self.input_type = np.float32
        self.other_array = other_array
        self.other_type = np.float32
        self._test(*self.create_model(), ie_device, precision, ir_version, use_convert_model=True)


class TestMulTypes(PytorchLayerTest):

    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randn(self.rhs_shape).to(self.rhs_type).numpy(),)
        elif len(self.rhs_shape) == 0:
            return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),)
        return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randn(self.rhs_shape).to(self.rhs_type).numpy())

    def create_model(self, lhs_type, lhs_shape, rhs_type, rhs_shape):

        class aten_mul(torch.nn.Module):
            def __init__(self, lhs_type, lhs_shape, rhs_type, rhs_shape):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type
                if len(lhs_shape) == 0:
                    self.forward = self.forward1
                elif len(rhs_shape) == 0:
                    self.forward = self.forward2
                else:
                    self.forward = self.forward3

            def forward1(self, rhs):
                return torch.mul(torch.tensor(3).to(self.lhs_type), rhs.to(self.rhs_type))

            def forward2(self, lhs):
                return torch.mul(lhs.to(self.lhs_type), torch.tensor(3).to(self.rhs_type))

            def forward3(self, lhs, rhs):
                return torch.mul(lhs.to(self.lhs_type), rhs.to(self.rhs_type))

        ref_net = None

        return aten_mul(lhs_type, lhs_shape, rhs_type, rhs_shape), ref_net, "aten::mul"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
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
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], []),
                                                          ([], [2, 3]),
                                                          ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_mul_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, lhs_shape, rhs_type, rhs_shape),
                   ie_device, precision, ir_version, freeze_model=False, trace_model=True)


class TestMulBool(PytorchLayerTest):
    def _prepare_input(self):
        input_tensor = torch.randint(0, 2, size=(1, 100)).numpy()
        other_tensor = torch.randint(0, 2, size=(1, 100)).numpy()
        return (input_tensor, other_tensor)

    def create_model(self, lhs_type, rhs_type):
        class aten_mul(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type

            def forward(self, input_tensor, other_tensor):
                return torch.mul(input_tensor.to(self.lhs_type), other_tensor.to(self.rhs_type))

        ref_net = None

        return aten_mul(), ref_net, "aten::mul"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"), [
        (torch.bool, torch.bool),
        (torch.bool, torch.int32),
        (torch.int32, torch.bool),
        (torch.float32, torch.bool),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_mul_bool(self, lhs_type, rhs_type, ie_device, precision, ir_version):
        self._test(*self.create_model(lhs_type, rhs_type), ie_device, precision, ir_version, use_convert_model=True)


class TestMulWithLhsComplex(PytorchLayerTest):
    def _prepare_input(self):
        rhs_input_shape = [3, 4, 5]
        lhs_input_shape = rhs_input_shape + [2]
        return [torch.randint(-10, 10, lhs_input_shape).to(self.lhs_type).numpy(),
                torch.randint(-10, 10, rhs_input_shape).to(self.rhs_type).numpy()]

    def create_model(self, op_type):
        class aten_mul(torch.nn.Module):

            def __init__(self, op) -> None:
                super().__init__()
                self.forward = self.forward1 if op == "mul" else self.forward2

            def forward1(self, lhs, rhs):
                lhs = torch.view_as_complex(lhs)
                res = torch.mul(lhs, rhs)
                return torch.view_as_real(res)

            def forward2(self, lhs, rhs):
                lhs = torch.view_as_complex(lhs)
                lhs.mul_(rhs)
                return torch.view_as_real(lhs)

        ref_net = None

        return aten_mul(op_type), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize("lhs_type",
                             [torch.float32,
                              torch.float64])
    @pytest.mark.parametrize("rhs_type",
                             [torch.int8,
                              torch.int32,
                              torch.int64,
                              torch.float32,
                              torch.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("op_type", ["mul", "mul_"])
    def test_mul(self, ie_device, precision, ir_version, lhs_type, rhs_type, op_type):
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type
        self._test(*self.create_model(op_type), ie_device, precision, ir_version,
                   use_convert_model=True)


class TestMulWithLhsComplex(PytorchLayerTest):
    def _prepare_input(self):
        rhs_input_shape = [3, 4, 5]
        lhs_input_shape = rhs_input_shape + [2]
        return [torch.randint(-10, 10, lhs_input_shape).to(self.lhs_type).numpy(),
                torch.randint(-10, 10, rhs_input_shape).to(self.rhs_type).numpy()]

    def create_model(self, op_type):
        class aten_mul(torch.nn.Module):

            def __init__(self, op) -> None:
                super().__init__()
                self.forward = self.forward1 if op == "mul" else self.forward2

            def forward1(self, lhs, rhs):
                lhs = torch.view_as_complex(lhs)
                res = torch.mul(lhs, rhs)
                return torch.view_as_real(res)

            def forward2(self, lhs, rhs):
                lhs = torch.view_as_complex(lhs)
                lhs.mul_(rhs)
                return torch.view_as_real(lhs)

        ref_net = None

        return aten_mul(op_type), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize("lhs_type",
                             [torch.float32,
                              torch.float64])
    @pytest.mark.parametrize("rhs_type",
                             [torch.int8,
                              torch.int32,
                              torch.int64,
                              torch.float32,
                              torch.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("op_type", ["mul", "mul_"])
    def test_mul(self, ie_device, precision, ir_version, lhs_type, rhs_type, op_type):
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type
        self._test(*self.create_model(op_type), ie_device, precision, ir_version,
                   use_convert_model=True)


class TestMulWithRhsComplex(PytorchLayerTest):
    def _prepare_input(self):
        lhs_input_shape = [3, 4, 5]
        rhs_input_shape = lhs_input_shape + [2]
        return [torch.randint(0, 10, lhs_input_shape).to(self.lhs_type).numpy(),
                torch.randint(0, 10, rhs_input_shape).to(self.rhs_type).numpy()]

    def create_model(self):
        class aten_mul(torch.nn.Module):

            def __init__(self) -> None:
                super().__init__()

            def forward(self, lhs, rhs):
                rhs = torch.view_as_complex(rhs)
                res = torch.mul(lhs, rhs)
                return torch.view_as_real(res)

        ref_net = None

        return aten_mul(), ref_net, f"aten::mul"

    @pytest.mark.parametrize("rhs_type",
                             [torch.float32,
                              torch.float64])
    @pytest.mark.parametrize("lhs_type",
                             [torch.uint8,
                              torch.int8,
                              torch.int32,
                              torch.int64,
                              torch.float32,
                              torch.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mul(self, ie_device, precision, ir_version, lhs_type, rhs_type):
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   use_convert_model=True)


class TestMulWithBothComplex(PytorchLayerTest):
    def _prepare_input(self):
        input_shape = [3, 4, 5]
        input_shape = input_shape + [2]
        return [torch.randint(0, 10, input_shape).to(self.lhs_type).numpy(),
                torch.randint(0, 10, input_shape).to(self.rhs_type).numpy()]

    def create_model(self, op_type):
        class aten_mul(torch.nn.Module):

            def __init__(self, op) -> None:
                super().__init__()
                self.forward = self.forward1 if op == "mul" else self.forward2

            def forward1(self, lhs, rhs):
                lhs = torch.view_as_complex(lhs)
                rhs = torch.view_as_complex(rhs)
                res = torch.mul(lhs, rhs)
                return torch.view_as_real(res)

            def forward2(self, lhs, rhs):
                lhs = torch.view_as_complex(lhs)
                rhs = torch.view_as_complex(rhs)
                res = lhs.mul_(rhs)
                return torch.view_as_real(res + lhs)

        ref_net = None

        return aten_mul(op_type), ref_net, f"aten::{op_type}"

    @pytest.mark.parametrize("lhs_type",
                             [torch.float32,
                              torch.float64])
    @pytest.mark.parametrize("rhs_type",
                             [torch.float32,
                              torch.float64])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("op_type", ["mul", "mul_"])
    def test_mul(self, ie_device, precision, ir_version, lhs_type, rhs_type, op_type):
        self.lhs_type = lhs_type
        self.rhs_type = rhs_type
        self._test(*self.create_model(op_type), ie_device, precision, ir_version,
                   use_convert_model=True)
