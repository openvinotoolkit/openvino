# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestAtenTo(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (np.random.randn(1, 3).astype(np.float32),)

    def create_model(self, type):

        import torch
        import torch.nn.functional as F

        class aten_to(torch.nn.Module):
            def __init__(self, type):
                super(aten_to, self).__init__()            
                self.type = type

            def forward(self, x):
                return x.to(self.type)

        ref_net = None

        return aten_to(type), ref_net

    @pytest.mark.parametrize("type", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.float32, torch.int64])
    @pytest.mark.nightly
    def test_aten_to(self, type, ie_device, precision, ir_version):
        if ie_device == "CPU":
            self._test(*self.create_model(type), ie_device, precision, ir_version)
