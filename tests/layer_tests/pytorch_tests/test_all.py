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

    @pytest.mark.parametrize("input_tensor", [
        np.eye(5,5),
        np.zeros((5, 5)),
        np.zeros((9,8)) + 1,
        np.random.randint(0, 2, (5, 9, 7)),
        np.random.randint(0, 2, (10, 13, 11)),
        np.random.randint(0, 2, (8, 7, 6, 5, 4)),
        np.random.randint(0, 2, (11, 11), dtype=np.uint8),
        np.random.randint(0, 2, (7, 7), dtype=np.uint8),
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_all_noparams(self, input_tensor, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        self._test(aten_all_noparam(), None, "aten::all", 
                ie_device, precision, ir_version, trace_model=True, freeze_model=False)
            
    @pytest.mark.parametrize("input_tensor", [
        np.eye(5,5),
        np.zeros((5, 5)),
        np.zeros((9,8)) + 1,
        np.random.randint(0, 2, (5, 9, 7)),
        np.random.randint(0, 2, (10, 13, 11)),
        np.random.randint(0, 2, (8, 7, 6, 5, 4)),
        np.random.randint(0, 2, (11, 11), dtype=np.uint8),
        np.random.randint(0, 2, (7, 7), dtype=np.uint8),
    ])
    @pytest.mark.parametrize("keepdim", [
        True,
        False,
        None
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_all(self, input_tensor, keepdim, ie_device, precision, ir_version):
        self.input_tensor = input_tensor
        for dim in range(len(input_tensor.shape)):
            self._test(aten_all(dim, keepdim), None, "aten::all", 
                    ie_device, precision, ir_version, trace_model=True, freeze_model=False)
