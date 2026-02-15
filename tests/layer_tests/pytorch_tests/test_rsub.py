# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestRsub(PytorchLayerTest):
    def _prepare_input(self):
        return self.input_data

    def create_model(self, second_type="float"):
        class aten_rsub_float(torch.nn.Module):

            def forward(self, x, y:float, alpha: float):
                return torch.rsub(x, y, alpha=alpha)

        class aten_rsub_int(torch.nn.Module):

            def forward(self, x, y:int, alpha: float):
                return torch.rsub(x, y, alpha=alpha)

        model_cls = {
            "float": aten_rsub_float,
            "int": aten_rsub_int
        }
        model = model_cls[second_type]


        ref_net = None

        return model(), ref_net, "aten::rsub"

    @pytest.mark.parametrize('input_data',
    [
        [[2, 3, 4], np.array(5).astype(np.float32), [1]]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_rsub1(self, ie_device, precision, ir_version, input_data):
        self.input_data = []
        for input in input_data:
            if type(input) is list:
                self.input_data.append(np.random.randn(*input).astype(np.float32))
            else:
                self.input_data.append(input)
        self._test(*self.create_model(second_type="float"), ie_device, precision, ir_version, use_convert_model=True)

    @pytest.mark.parametrize('input_data',
    [
        [[2, 3, 4], np.array(5).astype(int), [1]]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_rsub2(self, ie_device, precision, ir_version, input_data):
        self.input_data = []
        for input in input_data:
            if type(input) is list:
                self.input_data.append(np.random.randn(*input).astype(np.float32))
            else:
                self.input_data.append(input)
        self._test(*self.create_model(second_type="int"), ie_device, precision, ir_version, use_convert_model=True)


class TestRsubTypes(PytorchLayerTest):

    def _prepare_input(self):
        return (torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),
                np.array([1]).astype(self.rhs_type))

    def create_model(self, lhs_type, rhs_type):

        class aten_rsub(torch.nn.Module):
            def __init__(self, lhs_type, rhs_type):
                super().__init__()
                self.lhs_type = lhs_type
                if rhs_type == np.int32:
                    self.forward = self.forward2
                else:
                    self.forward = self.forward1

            def forward1(self, lhs, rhs:float):
                return torch.rsub(lhs.to(self.lhs_type), rhs, alpha=2)

            def forward2(self, lhs, rhs:int):
                return torch.rsub(lhs.to(self.lhs_type), rhs, alpha=2)

        ref_net = None

        return aten_rsub(lhs_type, rhs_type), ref_net, "aten::rsub"

    @pytest.mark.parametrize(("lhs_type", "rhs_type"),
                             [[torch.int32, np.int32],
                              [torch.int32, np.float32],
                              [torch.int64, np.int32],
                              [torch.int64, np.float32],
                              [torch.float32, np.int32],
                              [torch.float32, np.float32],
                              ])
    @pytest.mark.parametrize(("lhs_shape"), [[2, 3], [3], [2, 3, 4]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_rsub_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self._test(*self.create_model(lhs_type, rhs_type),
                   ie_device, precision, ir_version)
