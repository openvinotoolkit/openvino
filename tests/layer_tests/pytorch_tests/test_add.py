# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


@pytest.mark.parametrize('alpha', (-0.5, 0, 0.5, 1, 2))
@pytest.mark.parametrize('input_shape_rhs', [
    [2, 5, 3, 4],
    [1, 5, 3, 4],
    [1]
])
class TestAdd(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32), self.input_rhs)

    def create_model(self, alpha, op_type):
        class aten_add(torch.nn.Module):

            def __init__(self, alpha, op) -> None:
                super().__init__()
                self.alpha = alpha
                self.forward = self.forward1 if op == "add" else self.forward2

            def forward1(self, lhs, rhs):
                return torch.add(lhs, rhs, alpha=self.alpha)

            def forward2(self, lhs, rhs):
                return lhs.add_(rhs, alpha=self.alpha)

        ref_net = None

        return aten_add(alpha, op_type), ref_net, f"aten::{op_type}"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("op_type", ["add", skip_if_export("add_")])
    def test_add(self, ie_device, precision, ir_version, alpha, input_shape_rhs, op_type):
        self.input_rhs = np.random.randn(*input_shape_rhs).astype(np.float32)
        self._test(*self.create_model(alpha, op_type), ie_device, precision, ir_version, use_convert_model=True)


class TestAddTypes(PytorchLayerTest):

    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randint(0, 10, self.rhs_shape).to(self.rhs_type).numpy(),)
        elif len(self.rhs_shape) == 0:
            return (torch.randint(0, 10, self.lhs_shape).to(self.lhs_type).numpy(),)
        return (torch.randint(0, 10, self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randint(0, 10, self.rhs_shape).to(self.rhs_type).numpy())

    def create_model(self, lhs_type, lhs_shape, rhs_type, rhs_shape):

        class aten_add(torch.nn.Module):
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
                return torch.add(torch.tensor(1).to(self.lhs_type), rhs.to(self.rhs_type), alpha=2)

            def forward2(self, lhs):
                return torch.add(lhs.to(self.lhs_type), torch.tensor(1).to(self.rhs_type), alpha=2)

            def forward3(self, lhs, rhs):
                return torch.add(lhs.to(self.lhs_type), rhs.to(self.rhs_type), alpha=2)

        ref_net = None

        return aten_add(lhs_type, lhs_shape, rhs_type, rhs_shape), ref_net, "aten::add"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.bool, torch.uint8],
                              [torch.bool, torch.int8],
                              [torch.int8, torch.uint8],
                              [torch.uint8, torch.int8],
                              [torch.int32, torch.int64],
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
    @pytest.mark.precommit_fx_backend
    def test_add_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, lhs_shape, rhs_type, rhs_shape),
                   ie_device, precision, ir_version, freeze_model=False, trace_model=True)

class TestAddLists(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 5, 3, 4).astype(np.float32),)

    def create_model(self):
        class aten_add(torch.nn.Module):
            def forward(self, x):
                return x.reshape(x.shape[:-1] + (-1,))

        return aten_add(), None, "aten::add"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_add(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)


class TestAddBool(PytorchLayerTest):

    def _prepare_input(self):
        input2 = np.random.randint(0, 2, (1, 3, 20, 24)).astype(bool)
        input1 = np.random.randint(0, 2, (1, 3, 20, 24)).astype(bool)
        return (input1, input2)  

    def create_model(self, lhs_type=torch.bool, rhs_type=torch.bool):

        class aten_add(torch.nn.Module):
            def __init__(self):
                super(aten_add, self).__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type

            def forward(self, x1, x2):
                return torch.add(x1.to(self.rhs_type), x2.to(self.lhs_type))
        ref_net = None

        return aten_add(), ref_net, "aten::add"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"), [
        (torch.bool, torch.bool),
        (torch.bool, torch.int32),
        (torch.int32, torch.bool),
        (torch.float32, torch.bool),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_add(self, lhs_type, rhs_type, ie_device, precision, ir_version):
        self._test(*self.create_model(lhs_type, rhs_type), ie_device, precision, ir_version)
