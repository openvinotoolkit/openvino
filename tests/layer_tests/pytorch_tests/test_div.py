# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestDiv(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_array.astype(self.input_type), self.other_array.astype(self.other_type))

    def create_model(self, rounding_mode):

        class aten_div(torch.nn.Module):
            def __init__(self, rounding_mode):
                super(aten_div, self).__init__()
                self.rounding_mode = rounding_mode

            def forward(self, input_tensor, other_tensor):
                return torch.div(input_tensor, other_tensor, rounding_mode=self.rounding_mode)

        ref_net = None

        return aten_div(rounding_mode), ref_net, "aten::div"

    @pytest.mark.parametrize(("input_array", "other_array"), [
        [np.array([0.7620, 2.5548, -0.5944, -0.7438, 0.9274]), np.array(0.5)],
        [np.array([[-0.3711, -1.9353, -0.4605, -0.2917],
                   [0.1815, -1.0111, 0.9805, -1.5923],
                   [0.1062, 1.4581, 0.7759, -1.2344],
                   [-0.1830, -0.0313, 1.1908, -1.4757]]),
         np.array([0.8032, 0.2930, -0.8113, -0.2308])]
    ])
    @pytest.mark.parametrize('rounding_mode', ([
        None,
        "floor",
        "trunc"
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_div_pt_spec(self, input_array, other_array, rounding_mode, ie_device, precision, ir_version):
        self.input_array = input_array
        self.input_type = np.float32
        self.other_array = other_array
        self.other_type = np.float32
        self._test(*self.create_model(rounding_mode),
                   ie_device, precision, ir_version, use_convert_model=True)


class TestDivTypes(PytorchLayerTest):

    def _prepare_input(self):
        if len(self.lhs_shape) == 0:
            return (torch.randint(2, 5, self.rhs_shape).to(self.rhs_type).numpy(),)
        elif len(self.rhs_shape) == 0:
            return (10 * torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),)
        return (10 * torch.randn(self.lhs_shape).to(self.lhs_type).numpy(),
                torch.randint(2, 5, self.rhs_shape).to(self.rhs_type).numpy())

    def create_model(self, lhs_type, lhs_shape, rhs_type, rhs_shape, rounding_mode):

        class aten_div(torch.nn.Module):
            def __init__(self, lhs_type, lhs_shape, rhs_type, rhs_shape, rounding_mode):
                super().__init__()
                self.lhs_type = lhs_type
                self.rhs_type = rhs_type
                self.rm = rounding_mode
                if len(lhs_shape) == 0:
                    self.forward = self.forward1
                elif len(rhs_shape) == 0:
                    self.forward = self.forward2
                else:
                    self.forward = self.forward3

            def forward1(self, rhs):
                return torch.div(torch.tensor(3).to(self.lhs_type), rhs.to(self.rhs_type), rounding_mode=self.rm)

            def forward2(self, lhs):
                return torch.div(lhs.to(self.lhs_type), torch.tensor(3).to(self.rhs_type), rounding_mode=self.rm)

            def forward3(self, lhs, rhs):
                return torch.div(lhs.to(self.lhs_type), rhs.to(self.rhs_type), rounding_mode=self.rm)

        ref_net = None

        return aten_div(lhs_type, lhs_shape, rhs_type, rhs_shape, rounding_mode), ref_net, "aten::div"

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
    @pytest.mark.parametrize('rounding_mode', ([
        None,
        "floor",
        "trunc"
    ]))
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122715')
    def test_div_types(self, ie_device, precision, ir_version, lhs_type, lhs_shape, rhs_type, rhs_shape, rounding_mode):
        self.lhs_type = lhs_type
        self.lhs_shape = lhs_shape
        self.rhs_type = rhs_type
        self.rhs_shape = rhs_shape
        if rounding_mode == "floor" and not lhs_type.is_floating_point and not rhs_type.is_floating_point:
            pytest.skip("Floor rounding mode and int inputs produce wrong results")
        self._test(*self.create_model(lhs_type, lhs_shape, rhs_type, rhs_shape, rounding_mode),
                   ie_device, precision, ir_version)
