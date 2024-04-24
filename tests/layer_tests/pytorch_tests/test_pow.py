# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


@pytest.mark.parametrize('test_input,inplace', [
    ((np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[1, 1], [2, 2]], dtype=np.float32)), False),
    ((np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([2, 3], dtype=np.float32)), False),
    ((np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([2], dtype=np.float32)), False),
    ((np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[1, 1], [2, 2]], dtype=np.float32)), True),
    ((np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([2, 3], dtype=np.float32)), True),
    ((np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([2], dtype=np.float32)), True),
    ((np.array([5, 6], dtype=np.float32), np.array([[1, 2], [3, 4]], dtype=np.float32)), False),
    ((np.array([5], dtype=np.float32), np.array([[1, 2], [3, 4]], dtype=np.float32)), False),
    ])
class TestPow(PytorchLayerTest):
    """
    Input test data contains five test cases - elementwise power, broadcast exponent, one exponent,
    broadcast base, one base.
    """

    def _prepare_input(self):
        return self.test_input

    def create_model(self, inplace):
        class aten_pow(torch.nn.Module):
            def __init__(self, inplace):
                super(aten_pow, self).__init__()
                if inplace:
                    self.forward = self.forward_inplace
                else:
                    self.forward = self.forward_

            def forward_(self, input_data, exponent):
                return torch.pow(input_data, exponent)

            def forward_inplace(self, input_data, exponent):
                return input_data.pow_(exponent)

        return aten_pow(inplace), None, "aten::pow_" if inplace else "aten::pow"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_pow(self, inplace, ie_device, precision, ir_version, test_input):
        if inplace and PytorchLayerTest.use_torch_export():
            pytest.skip(reason="export fails for inplace")
        self.test_input = test_input
        self._test(*self.create_model(inplace), ie_device, precision,
                   ir_version, use_convert_model=True)


class TestPowMixedTypes(PytorchLayerTest):
    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randn(self.rhs_shape) * 2 + 0.6).to(self.rhs_type).numpy(),
        elif len(self.rhs_shape) == 0:
            return (torch.randint(1, 3, self.lhs_shape).to(self.lhs_type).numpy(),)
        return (torch.randint(1, 3, self.lhs_shape).to(self.lhs_type).numpy(),
                (torch.randn(self.rhs_shape) * 2 + 0.6).to(self.rhs_type).numpy())

    def create_model(self, lhs_type, lhs_shape, rhs_type, rhs_shape):

        class aten_pow(torch.nn.Module):
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
                return torch.pow(torch.tensor(3).to(self.lhs_type), rhs.to(self.rhs_type))

            def forward2(self, lhs):
                return torch.pow(lhs.to(self.lhs_type), torch.tensor(3).to(self.rhs_type))

            def forward3(self, lhs, rhs):
                return torch.pow(lhs.to(self.lhs_type), rhs.to(self.rhs_type))

        ref_net = None

        return aten_pow(lhs_type, lhs_shape, rhs_type, rhs_shape), ref_net, "aten::pow"

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
                              ])
    @pytest.mark.parametrize(("lhs_shape", "rhs_shape"), [([2, 3], [2, 3]),
                                                          ([2, 3], []),
                                                          ([], [2, 3]),
                                                          ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_pow_mixed_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        if ie_device == "GPU" and rhs_type not in [torch.float32, torch.float64] and lhs_type not in [torch.float32, torch.float64]:
            pytest.xfail(reason="pow is not supported on GPU for integer types")
        self._test(*self.create_model(lhs_type, lhs_shape, rhs_type, rhs_shape),
                   ie_device, precision, ir_version)


class TestPowMixedTypesScalars(PytorchLayerTest):
    def _prepare_input(self):
        return (torch.randn([1, 2, 3, 4]).numpy(),)

    def create_model(self):

        class aten_pow(torch.nn.Module):
            def forward(self, x):
                return torch.pow(x.size(2), -0.5)

        ref_net = None

        return aten_pow(), ref_net, "aten::pow"

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_pow_mixed_types(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version)
