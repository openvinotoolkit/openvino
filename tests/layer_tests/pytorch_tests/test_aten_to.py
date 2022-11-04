# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestAtenTo(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1, 3).astype(self.input_type),)

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

    # Cartesian product of input/output types
    @pytest.mark.parametrize("input_type", [np.int32, np.float32, np.float64])
    @pytest.mark.parametrize("output_type", [torch.uint8, torch.int8, torch.int16, torch.int32, torch.float32, torch.int64])
    @pytest.mark.nightly
    def test_aten_to(self, input_type, output_type, ie_device, precision, ir_version):
            self.input_type = input_type
            self._test(*self.create_model(output_type), ie_device, precision, ir_version)
