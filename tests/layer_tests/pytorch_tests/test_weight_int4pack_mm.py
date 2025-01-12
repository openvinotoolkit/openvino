import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest

class TestWeightInt4PackMMOperation(PytorchLayerTest):
    def _prepare_input(self, input_size):
        return (torch.rand(1,input_size,dtype=torch.float32),)

    def create_model(self, input_size, group_size, inner_k_tiles):
        class CustomWeightInt4PackMMOperation(torch.nn.Module):
            def __init__(self):
                super(CustomWeightInt4PackMMOperation, self).__init__()
                # TODO: Generating random quantized weights, scales, and zero points is not ideal.
                # An actual quantizing algorithm could be implemented for more accurate testing.
                self.gs = group_size
                w = torch.randint(low=0,high=15,size=[input_size,input_size], dtype=torch.int32)
                self.wq = torch.ops.aten._convert_weight_to_int4pack(w, inner_k_tiles)
                scales = torch.randint(low=1,high=100,size=[int(input_size/self.gs),input_size,1], dtype=torch.int32).to(dtype=torch.bfloat16)/10.0
                zeros = torch.ones(int(input_size/self.gs),input_size,1, dtype=torch.bfloat16)*8.0
                self.sz = torch.cat((scales,zeros), 2)
            def forward(self, x):
                return torch.ops.aten._weight_int4pack_mm(x.to(dtype=torch.bfloat16), self.wq, self.gs, self.sz).to(dtype=torch.float32)

        model_class = CustomWeightInt4PackMMOperation()
        ref_net = None
        return model_class, ref_net, "aten._weight_int4pack_mm.default"  

    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("input_size, group_size, inner_k_tiles, dtype", [
        (1024, 32, 2, torch.float32),
        (1024, 32, 4, torch.float32),
        (1024, 32, 8, torch.float32),
        (4096, 32, 2, torch.float32),
        (4096, 32, 4, torch.float32),
        (4096, 32, 8, torch.float32),
        (4096, 64, 2, torch.float32),
        (4096, 64, 4, torch.float32),
        (4096, 64, 8, torch.float32),
    ])
    def test_weight_int4pack_mm_operation(self, input_size, group_size, inner_k_tiles, dtype, ie_device, precision, ir_version):
        # Due to precision errors, the output accuracy may change based on the system this test it running on.
        # The eps is adjusted accordingly, but overall model accuracy should be observed in full model tests as well.
        self._test(
            *self.create_model(input_size, group_size, inner_k_tiles),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"input_size": input_size},
            aot_autograd=True,
            custom_eps=128.0
        )
