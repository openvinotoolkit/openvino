# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest


class TestRot90(PytorchLayerTest):
    def _prepare_input(self):

        x = np.arange(24).reshape(2, 3, 4).astype(np.float32)
        return (x,)
    
    def create_model(self, k, dims):
        import torch

        class aten_rot90(torch.nn.Module):
            def __init__(self, k=1, dims=(0, 1)):
                super(aten_rot90, self).__init__()
                self.k = k
                self.dims = dims

            def forward(self, x):
                return torch.rot90(x, self.k, self.dims)
                
        ref_net = None
        return aten_rot90(k, dims), ref_net, "aten::rot90"
    
    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5])
    @pytest.mark.parametrize("dims", [(0, 1), (0, 2), (1, 2)])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_rot90(self, k, dims, ie_device, precision, ir_version):
        self._test(*self.create_model(k, dims), ie_device, precision, ir_version, 
                   trace_model=True,dynamic_shapes=False)    