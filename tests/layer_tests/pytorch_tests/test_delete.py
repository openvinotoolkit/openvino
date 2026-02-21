
import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestDelete(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(2, 3).astype(np.float32),)

    class ListDeleteModel(torch.nn.Module):
        def forward(self, x):
            l = [x, x + 1.0]
            del l[-1]
            # aten::Delete should be ignored in graph
            return l[0]

    class DictDeleteModel(torch.nn.Module):
        def forward(self, x):
            d = {"a": x, "b": x + 1.0}
            del d["a"]
            # aten::Delete should be ignored
            return d["b"]

    @pytest.mark.parametrize("model_class", [
        ListDeleteModel,
        DictDeleteModel
    ])
    def test_delete(self, model_class, ie_device, precision, ir_version):
        model = model_class()
        # We verify that deletion ops don't crash the conversion
        self._test(model, None, "aten::add", 
                   ie_device, precision, ir_version)
