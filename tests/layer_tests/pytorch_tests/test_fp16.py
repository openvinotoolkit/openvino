# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestBF16(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(10).astype(np.float32),)

    def create_model(self):
        class aten_add(torch.nn.Module):
            def __init__(self):
                super(aten_add, self).__init__()
                self.y = torch.randn(10, dtype=torch.bfloat16)

            def forward(self, x):
                return x + self.y.to(torch.float32)

        return aten_add(), None, "aten::add"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("to_trace", [True, False])
    def test_bf16(self, ie_device, precision, ir_version, to_trace):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=to_trace, freeze_model=False, use_convert_model=True)


class TestFP16(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(10).astype(np.float32),)

    def create_model(self):
        class aten_add(torch.nn.Module):
            def __init__(self):
                super(aten_add, self).__init__()
                self.y = torch.randn(10, dtype=torch.float16)

            def forward(self, x):
                return x + self.y.to(torch.float32)

        return aten_add(), None, "aten::add"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("to_trace", [True, False])
    def test_fp16(self, ie_device, precision, ir_version, to_trace):
        self._test(*self.create_model(), ie_device, precision,
                   ir_version, trace_model=to_trace, freeze_model=False, use_convert_model=True)
