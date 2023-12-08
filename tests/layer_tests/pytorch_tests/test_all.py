# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_all_noparam(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor):
        return torch.all(input_tensor)

class aten_all(torch.nn.Module):
    def __init__(self, dim, keepdim) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input_tensor):
        return torch.all(
            input_tensor,
            dim = self.dim
        ) if self.keepdim is None else torch.all(
            input_tensor,
            dim = self.dim,
            keepdim = self.keepdim
        )

class TestAll(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.parametrize("input_shape, d_type", [
        (np.eye(5,5), np.int64),
        (np.zeros((5, 5)), np.int64),
        (np.zeros((9,8)) + 1, np.int64),
        ([5, 9, 7], np.int64),
        ([10, 13, 11], np.int64),
        ([8, 7, 6, 5, 4], np.int64),
        ([11, 11], np.uint8),
        ([7, 7], np.uint8)
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_all_noparams(self, input_shape, d_type, ie_device, precision, ir_version):
        if type(input_shape) is list:
            self.input_tensor = np.random.randint(0, 2, input_shape, dtype=d_type)
        else:
            self.input_tensor = input_shape
        self._test(aten_all_noparam(), None, "aten::all",
                ie_device, precision, ir_version, trace_model=True, freeze_model=False)

    @pytest.mark.parametrize("input_shape, d_type", [
        (np.eye(5,5), np.int64),
        (np.zeros((5, 5)), np.int64),
        (np.zeros((9,8)) + 1, np.int64),
        ([5, 9, 7], np.int64),
        ([10, 13, 11], np.int64),
        ([8, 7, 6, 5, 4], np.int64),
        ([11, 11], np.uint8),
        ([7, 7], np.uint8)
    ])
    @pytest.mark.parametrize("keepdim", [
        True,
        False,
        None
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122715')
    def test_all(self, input_shape, d_type, keepdim, ie_device, precision, ir_version):
        if type(input_shape) is list:
            self.input_tensor = np.random.randint(0, 2, input_shape, dtype=d_type)
        else:
            self.input_tensor = input_shape
        for dim in range(len(self.input_tensor.shape)):
            self._test(aten_all(dim, keepdim), None, "aten::all",
                    ie_device, precision, ir_version, trace_model=True, freeze_model=False)
