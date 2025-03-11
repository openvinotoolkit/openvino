# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_layer_test_class import PytorchLayerTest, skip_if_export

class TestNanToNum(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([float('nan'), float('inf'), -float('inf'), 1.0, -1.0, 0.0], dtype=np.float32),)

    def create_model(self, nan_replacement, posinf_replacement, neginf_replacement, inplace):
        class aten_nan_to_num(torch.nn.Module):
            def __init__(self, nan_replacement, posinf_replacement, neginf_replacement, inplace):
                super(aten_nan_to_num, self).__init__()
                self.nan_replacement = nan_replacement
                self.posinf_replacement = posinf_replacement
                self.neginf_replacement = neginf_replacement
                self.inplace = inplace

            def forward(self, x):
                return x, torch.nan_to_num(x, nan=self.nan_replacement, posinf=self.posinf_replacement, neginf=self.neginf_replacement, inplace=self.inplace)

        ref_net = None
        return aten_nan_to_num(nan_replacement, posinf_replacement, neginf_replacement, inplace), ref_net, "aten::nan_to_num" if not inplace else "aten::nan_to_num_"

    @pytest.mark.parametrize("nan_replacement", [0.0, 1.0, -1.0])
    @pytest.mark.parametrize("posinf_replacement", [float('inf'), 3.0, 100.0])
    @pytest.mark.parametrize("neginf_replacement", [-float('inf'), -3.0, -100.0])
    @pytest.mark.parametrize("inplace", [skip_if_export(True), False])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    def test_nan_to_num(self, nan_replacement, posinf_replacement, neginf_replacement, inplace, ie_device, precision, ir_version):
        self._test(*self.create_model(nan_replacement, posinf_replacement, neginf_replacement, inplace), ie_device, precision, ir_version)

    @pytest.mark.parametrize("nan_replacement", [0.0, 1.0, -1.0])
    @pytest.mark.parametrize("posinf_replacement", [float('inf'), 3.0, 100.0])
    @pytest.mark.parametrize("neginf_replacement", [-float('inf'), -3.0, -100.0])
    def test_nan_to_num_inplace(self, nan_replacement, posinf_replacement, neginf_replacement, ie_device, precision, ir_version):
        self._test(*self.create_model(nan_replacement, posinf_replacement, neginf_replacement, True), ie_device, precision, ir_version)

    # Test case for integer inputs (should return as-is)
    @pytest.mark.parametrize("dtype", [np.int32, np.int64])
    def test_nan_to_num_integer(self, dtype, ie_device, precision, ir_version):
        x = np.array([1, 2, 3, 0, -1], dtype=dtype)
        model, ref_net, op_name = self.create_model(0, 100, -100, False)
        self._test(model, ref_net, ie_device, precision, ir_version)

    def test_nan_to_num_convertlike(self, ie_device, precision, ir_version):
        x = torch.tensor([float('nan'), float('inf'), -float('inf')], dtype=torch.float16)
        model = self.create_model(torch.tensor(0, dtype=torch.float32),
                                  torch.tensor(1e4, dtype=torch.float32),
                                  torch.tensor(-1e4, dtype=torch.float32), False)
        self._test(*model, ie_device, precision, ir_version)

    @pytest.mark.parametrize("x", [
        np.array([], dtype=np.float32), 
        np.array([float('nan')] * 10, dtype=np.float32),  
        np.array([float('inf')] * 10, dtype=np.float32),  
        np.array([-float('inf')] * 10, dtype=np.float32),  
        np.array([1e38, -1e38, 0.0, float('nan'), float('inf'), -float('inf')], dtype=np.float32),  
    ])
    def test_nan_to_num_edge_cases(self, x, ie_device, precision, ir_version):
        model, ref_net, op_name = self.create_model(0.0, 1e6, -1e6, False)
        self._test(model, ref_net, ie_device, precision, ir_version)

