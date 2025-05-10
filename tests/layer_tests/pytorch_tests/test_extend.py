import pytest
import torch
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest

class TestExtendList(PytorchLayerTest):
    def _prepare_input(self):
        return (np.random.randn(1).astype(np.float32),)

    def create_model(self, num1, num2, num3, num4, num5):
        class AtenExtendModel(torch.nn.Module):
            def __init__(self, num1, num2, num3, num4, num5):
                super().__init__()
                self.num1 = num1
                self.num2 = num2
                self.num3 = num3
                self.num4 = num4
                self.num5 = num5

            def forward(self, x):
                list1 = [self.num1, self.num2, self.num3]
                list2 = [self.num4, self.num5]

                list1.extend(list2)
                return torch.tensor(list1)

        ref_net = None
        expected_ops = ["prim::ListConstruct", "aten::extend"]

        return AtenExtendModel(num1, num2, num3, num4, num5), ref_net, expected_ops

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("num1,num2,num3,num4,num5", [(1, 1, 1, 1, 1), 
                                                          (1.5, 1.5, 1.5, 1.5, 1.5), 
                                                          (1, 2, 3, 4, 5), 
                                                          (4.7, 346.5, 24.6, 34.6, 334.6)])
    def test_extend_list_constants(self, num1, num2, num3, num4, num5, ie_device, precision, ir_version):
        self._test(
            *self.create_model(num1, num2, num3, num4, num5),
            ie_device,
            precision,
            ir_version
            # No kwargs_to_prepare_input needed as input is simple dummy data
        )
