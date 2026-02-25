# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestFloorDivide(PytorchLayerTest):
    rng = np.random.default_rng(seed=123)

    def _prepare_input(self):
        return (self.input_tensor, self.other_tensor)

    def create_model(self):
        import torch

        class aten_floor_divide(torch.nn.Module):
            def __init__(self):
                super(aten_floor_divide, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.floor_divide(input_tensor, other_tensor)

        return aten_floor_divide(), None, "aten::floor_divide"

    def create_model_int(self):
        import torch

        class aten_floor_divide(torch.nn.Module):
            def __init__(self):
                super(aten_floor_divide, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return torch.floor_divide(input_tensor.to(torch.int32), other_tensor.to(torch.int64))

        return aten_floor_divide(), None, "aten::floor_divide"

    def create_model_inplace(self):
        import torch

        class aten_floor_divide_(torch.nn.Module):
            def __init__(self):
                super(aten_floor_divide_, self).__init__()

            def forward(self, input_tensor, other_tensor):
                return input_tensor.floor_divide_(other_tensor), input_tensor

        return aten_floor_divide_(), None, "aten::floor_divide_"

    @pytest.mark.parametrize('input_tensor', [
        [5], [5, 5, 1], [1, 1, 5, 5],
    ])
    @pytest.mark.parametrize('other_tensor', [
        np.array([[0.5]]).astype(np.float32), [5], [5, 1], [1, 5]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_floor_divide(self, input_tensor, other_tensor, ie_device, precision, ir_version):
        if isinstance(input_tensor, list):
            self.input_tensor = self.rng.standard_normal(input_tensor, dtype=np.float32)
        else:
            self.input_tensor = input_tensor
        if isinstance(other_tensor, list):
            self.other_tensor = self.rng.standard_normal(other_tensor, dtype=np.float32)
        else:
            self.other_tensor = other_tensor
        self._test(*self.create_model(), ie_device, precision, ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.parametrize('input_tensor', [
        [5, 5, 5], [1, 1, 5, 5],
    ])
    @pytest.mark.parametrize('other_tensor', [
        np.array([0.5]).astype(np.float32), [5], [5, 1], [1, 5]
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_floor_divide_(self, input_tensor, other_tensor, ie_device, precision, ir_version):
        if isinstance(input_tensor, list):
            self.input_tensor = self.rng.standard_normal(input_tensor, dtype=np.float32)
        else:
            self.input_tensor = input_tensor
        if isinstance(other_tensor, list):
            self.other_tensor = self.rng.standard_normal(other_tensor, dtype=np.float32)
        else:
            self.other_tensor = other_tensor
        self._test(*self.create_model_inplace(), ie_device, precision, ir_version, trace_model=True, use_convert_model=True)

    @pytest.mark.parametrize('input_data', [
        {"tensor": [5], "low": 0, "high": 10},
        {"tensor": [5, 5, 1], "low": 1, "high": 10},
        {"tensor": [1, 1, 5, 5], "low": 1, "high": 10}
    ])
    @pytest.mark.parametrize('other_data', [
        {"tensor": np.array([[2]]).astype(np.float32)},
        {"tensor": [5], "low": 1, "high": 10},
        {"tensor": [5, 1], "low": 1, "high": 10},
        {"tensor":  [5, 1], "low": 1, "high": 10}
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_floor_divide_int(self, input_data, other_data, ie_device, precision, ir_version):
        input_tensor = input_data["tensor"]
        if isinstance(input_tensor, list):
            self.input_tensor = self.rng.integers(low=input_data["low"],
                                                  high=input_data["high"],
                                                  size=input_tensor).astype(np.float32)
        else:
            self.input_tensor = input_tensor

        other_tensor = other_data["tensor"]
        if isinstance(other_tensor, list):
            self.other_tensor = self.rng.integers(low=other_data["low"],
                                                  high=other_data["high"],
                                                  size=other_tensor).astype(np.float32)
        else:
            self.other_tensor = other_tensor
        self._test(*self.create_model_int(), ie_device, precision, ir_version)
