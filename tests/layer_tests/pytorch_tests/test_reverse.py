# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class aten_reverse(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.tensor(1, dtype=x.dtype)
        b = torch.tensor(2, dtype=x.dtype)
        lst = [a, b]
        lst.reverse()
        return torch.stack(lst)


class aten_reverse_out(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        a = torch.tensor(1, dtype=x.dtype)
        b = torch.tensor(2, dtype=x.dtype)
        lst = [a, b]
        lst.reverse()
        return torch.stack(lst), torch.tensor(len(lst), dtype=torch.int64)


class TestReverse(PytorchLayerTest):
    def _prepare_input(self):
        data = np.random.randn(2, 3).astype(np.float32)
        return (data,)

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_no_out(self, ie_device, precision, ir_version):
        model = aten_reverse()
        self._test(model, None, "aten::reverse", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={})

    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_reverse_with_out(self, ie_device, precision, ir_version):
        model = aten_reverse_out()
        self._test(model, None, "aten::reverse", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={})
