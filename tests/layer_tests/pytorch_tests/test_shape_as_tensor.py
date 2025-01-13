# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest

class aten_shape_as_tensor(torch.nn.Module):
    def __init__(self) -> None:
        torch.nn.Module.__init__(self)

    def forward(self, input_tensor):
        return torch.ops.aten._shape_as_tensor(input_tensor)

class TestShapeAsTensor(PytorchLayerTest):
    def _prepare_input(self):
        return (self.input_tensor,)

    @pytest.mark.parametrize("shape", [
        # (),
        (2,),
        (1,2,3,4), 
        (5,4,2,7)
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_all_noparams(self, shape, ie_device, precision, ir_version):
        self.input_tensor = np.zeros(shape)
        self._test(aten_shape_as_tensor(), None, "aten::_shape_as_tensor", 
                ie_device, precision, ir_version)
