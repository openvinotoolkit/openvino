# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
  
import pytest  
import torch  
import numpy as np  
from pytorch_layer_test_class import PytorchLayerTest  
  
  
class TestAutoCastToReducedPrecision(PytorchLayerTest):  
    def _prepare_input(self, input_shape):    
        return (self.random.randn(*input_shape),) 
  
    def create_model(self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype):  
        class aten_autocast_to_reduced_precision(torch.nn.Module):  
            def __init__(self, cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype):  
                super().__init__()  
                self.cuda_enabled = cuda_enabled  
                self.cpu_enabled = cpu_enabled  
                self.cuda_dtype = cuda_dtype  
                self.cpu_dtype = cpu_dtype  
  
            def forward(self, x):  
                return torch.ops.aten._autocast_to_reduced_precision(  
                    x, self.cuda_enabled, self.cpu_enabled,  
                    self.cuda_dtype, self.cpu_dtype  
                ).to(torch.float32)  
  
        return (  
            aten_autocast_to_reduced_precision(cuda_enabled, cpu_enabled, cuda_dtype, cpu_dtype),  
            "aten::_autocast_to_reduced_precision"  
        )  
  
    @pytest.mark.nightly  
    @pytest.mark.precommit  
    def test_autocast_to_reduced_precision(self, ie_device, precision, ir_version):  
        cuda_dtype = torch.float16  
        cpu_dtype = torch.bfloat16  
        self._test(  
            *self.create_model(False, False, cuda_dtype, cpu_dtype),  
            ie_device, precision, ir_version,  
            kwargs_to_prepare_input={"input_shape": [2, 3, 4]}  
        )
