# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import numpy as np
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class aten_hstack(torch.nn.Module):
    def forward(self, x):
        return torch.hstack(self.prepare_input(x))
    
    def prepare_input(self, x):
        return (x, x)
    
class aten_hstack_out(aten_hstack):
    def forward(self, x, out):
        return torch.hstack(self.prepare_input(x), out=out), out
    
class TestHstack(PytorchLayerTest):
    def _prepare_input(self, out=False, num_repeats=2):
        data = np.random.randn(2, 1, 3)
        if not out:
            return (data, )
        concat = [data for _ in range(num_repeats)]
        out = np.zeros_like(np.concatenate(concat, axis=1))
        return (data, out)
    
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("out", [False, True])
    def test_hstack(self, out, ie_device, precision, ir_version):
        model = aten_hstack() if not out else aten_hstack_out()
        self._test(model, None, "aten::hstack", ie_device, 
                   precision, ir_version, kwargs_to_prepare_input={"out": out, "num_repeats": 2})
        
    
class TestHstackAlignTypes(PytorchLayerTest):
    def _prepare_input(self, in_types):
        in_vals = []
        for i in range(len(in_types)):
            dtype = in_types[i]
            in_vals.append(np.random.randn(2, 1, 3).astype(dtype))
        return in_vals
    
    def create_model(self, in_count):
        class aten_align_types_hstack_two_args(torch.nn.Module):            
            def forward(self, x, y):
                ins = [x, y]
                return torch.hstack(ins)
        
        class aten_align_types_hstack_three_args(torch.nn.Module):
            def forward(self, x, y, z):
                ins = [x, y, z]
                return torch.hstack(ins)
            
        if in_count == 2:
            return aten_align_types_hstack_two_args()
        
        if in_count == 3:
            return aten_align_types_hstack_three_args()
        
    @pytest.mark.parametrize(("in_types"), [
        (np.float32, np.int32),
        (np.int32, np.float32),
        (np.float16, np.float32),
        (np.int16, np.float16),
        (np.int32, np.int64),
        # # Three inputs
        (np.float32, np.int32, np.int32),
        (np.float32, np.int32, np.float32),
        (np.int32, np.float32, np.int32),
        (np.float32, np.int32, np.int16),
        (np.int32, np.float32, np.int16),
        (np.int16, np.int32, np.int16),
        (np.float16, np.float32, np.float16),
        (np.float32, np.float16, np.float32),
        (np.float16, np.int32, np.int16),
        (np.int16, np.float16, np.int16)
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_align_types_hstack(self, ie_device, precision, ir_version, in_types):
        self._test(self.create_model(len(in_types)), None, "aten::hstack",
                   ie_device, precision, ir_version, kwargs_to_prepare_input={"in_types": in_types})