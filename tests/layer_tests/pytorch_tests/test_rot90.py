import torch
import numpy as np
import pytest

from pytorch_layer_test_class import PytorchLayerTest

class TestRot90(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.uniform(0, 50, (2, 3, 4)).astype(np.float32),) 
    def create_model(self,k,dims):
        class aten_rot90(torch.nn.Module):
            def __init__(self, k, dims=(0,1)):
                super(aten_rot90, self).__init__()
                self.k = k
                self.dims = dims
            def forward(self,x):
                return torch.rot90(x,self.k,self.dims)
        ref_net = None
        return aten_rot90(k,dims), ref_net, "aten::rot90"
    
    @pytest.mark.parametrize('k',[
       1,2,3,4,-1,-3
    ])
    @pytest.mark.parametrize('dims',[
        (0,1),(1,2),(0,2),(-2,-3),(-1,1),(-1,-3)
    ])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    
    def test_rot_90(self,k,dims,ie_device,precision,ir_version):
        self._test(*self.create_model(k,dims),ie_device, precision, ir_version,
                   dynamic_shapes=ie_device != "GPU")
