# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestSub(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_data

    def create_model(self, inplace):

        class aten_sub(torch.nn.Module):
            def __init__(self, inplace) -> None:
                super().__init__()
                if inplace:
                    self.forward = self._forward_inplace
                else:
                    self.forward = self._forward_out_of_place

            def _forward_out_of_place(self, x, y, alpha: float):
                return torch.sub(x, y, alpha=alpha)

            def _forward_inplace(self, x, y, alpha: float):
                return x.sub_(y, alpha=alpha)

        ref_net = None

        if inplace:
            op_name = "aten::sub_"
        else:
            op_name = "aten::sub"

        return aten_sub(inplace), ref_net, op_name

    @pytest.mark.parametrize('input_shapes',
    [
        [
            [2, 3, 4], [2, 3, 4], [1]
        ],
        [
            [4, 2, 3], [1, 2, 3], [1]
        ]
    ])
    @pytest.mark.parametrize("inplace", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sub(self, ie_device, precision, ir_version, input_shapes, inplace):
        self.input_data = []
        for input_shape in input_shapes:
            self.input_data.append(np.random.random_sample(input_shape).astype(np.float32))
        self._test(*self.create_model(inplace), ie_device, precision, ir_version, use_convert_model=True)


class TestSubTypes(PytorchLayerTest):

    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randn(self.rhs_shape).to(self.rhs_type).numpy(),)
        elif len(self.rhs_shape) == 0:
            return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),)
        return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randn(self.rhs_shape).to(self.rhs_type).numpy())

    def create_model(self, lhs_type, lhs_shape, rhs_type, rhs_shape):

        class aten_sub(torch.nn.Module):
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
                return torch.sub(torch.tensor(3).to(self.lhs_type), rhs.to(self.rhs_type), alpha=2)

            def forward2(self, lhs):
                return torch.sub(lhs.to(self.lhs_type), torch.tensor(3).to(self.rhs_type), alpha=2)

            def forward3(self, lhs, rhs):
                return torch.sub(lhs.to(self.lhs_type), rhs.to(self.rhs_type), alpha=2)

        ref_net = None

        return aten_sub(lhs_type, lhs_shape, rhs_type, rhs_shape), ref_net, "aten::sub"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.int32, torch.int64],
                              [torch.int32, torch.float32],
                              # [torch.int32, torch.float64], fp64 produce ov error of eltwise constant fold
                              [torch.int64, torch.int32],
                              [torch.int64, torch.float32],
                              # [torch.int64, torch.float64], fp64 produce ov error of eltwise constant fold
                              [torch.float32, torch.int32],
                              [torch.float32, torch.int64],
                              # [torch.float32, torch.float64], fp64 produce ov error of eltwise constant fold
                              ])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], []),
                                                          ([], [2, 3]),
                                                          ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_sub_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        self._test(*self.create_model(lhs_type, lhs_shape, rhs_type, rhs_shape),
                   ie_device, precision, ir_version)
