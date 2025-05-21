# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export

class aten_any_noparam(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor):
        return torch.any(input_tensor)

class aten_any_noparam_out(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor, out):
        return torch.any(input_tensor, out=out), out

class aten_any(torch.nn.Module):
    def __init__(self, dim, keepdim) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input_tensor):
        return torch.any(
            input_tensor,
            dim = self.dim
        ) if self.keepdim is None else torch.any(
            input_tensor,
            dim = self.dim,
            keepdim = self.keepdim
        )

class aten_any_out(torch.nn.Module):
    def __init__(self, dim, keepdim) -> None:
        torch.nn.Module.__init__(self)
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, input_tensor, out):
        return torch.any(
            input_tensor,
            dim = self.dim,
            out=out
        ) if self.keepdim is None else torch.any(
            input_tensor,
            dim = self.dim,
            keepdim = self.keepdim,
            out=out
        ), out

class TestAny(PytorchLayerTest):
    def _prepare_input(self, out=False):
        if not out:
            return (self.input_tensor,)
        return (self.input_tensor, np.zeros_like(self.input_tensor, dtype=bool if self.input_tensor.dtype != np.uint8 else np.uint8))

    @pytest.mark.parametrize("input_shape, d_type", [
        (np.eye(5,5), np.int64),
        (np.zeros((5, 5)), np.int64),
        (np.zeros((9,8)) + 1, np.int64),
        ([5, 9, 7], np.int64),
        ([10, 13, 11], np.int64),
        ([8, 7, 6, 5, 4], np.int64),
        ([11, 11], np.uint8),
        ([7, 7], np.uint8),
        ([4, 4], bool)
    ])
    @pytest.mark.parametrize("out", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    def test_any_noparams(self, input_shape, d_type, out, ie_device, precision, ir_version):
        if type(input_shape) is list:
            self.input_tensor = np.random.randint(0, 2, input_shape, dtype=d_type)
        else:
            self.input_tensor = input_shape
        self._test(aten_any_noparam() if not out else aten_any_noparam_out(), None, "aten::any",
                ie_device, precision, ir_version, trace_model=True, freeze_model=False, kwargs_to_prepare_input={"out": out})

    @pytest.mark.parametrize("input_shape, d_type", [
        (np.eye(5,5), np.int64),
        (np.zeros((5, 5)), np.int64),
        (np.zeros((9,8)) + 1, np.int64),
        ([5, 9, 7], np.int64),
        ([10, 13, 11], np.int64),
        ([8, 7, 6, 5, 4], np.int64),
        ([11, 11], np.uint8),
        ([7, 7], np.uint8),
        ([4, 4], bool)
    ])
    @pytest.mark.parametrize("keepdim", [
        True,
        False,
        None
    ])
    @pytest.mark.parametrize("out", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 122715')
    def test_any(self, input_shape, d_type, keepdim, out, ie_device, precision, ir_version):
        if type(input_shape) is list:
            self.input_tensor = np.random.randint(0, 2, input_shape, dtype=d_type)
        else:
            self.input_tensor = input_shape
        for dim in range(len(self.input_tensor.shape)):
            self._test(aten_any(dim, keepdim) if not out else aten_any_out(dim, keepdim), None, "aten::any",
                    ie_device, precision, ir_version, trace_model=True, freeze_model=False, kwargs_to_prepare_input={"out": out})
