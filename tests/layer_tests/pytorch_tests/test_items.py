import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestItems(PytorchLayerTest):
    def _prepare_input(self):
        return []

    def create_model(self, dict_type="int_int"):
        class AtenItemsIntInt(torch.nn.Module):
            def forward(self):
                d = {1: 10, 2: 20}
                return d.items()

        class AtenItemsStrTensor(torch.nn.Module):
            def forward(self):
                d = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
                return d.items()

        model_map = {
            "int_int": AtenItemsIntInt(),
            "str_tensor": AtenItemsStrTensor(),
        }

        return model_map[dict_type], None, "aten::items"

    @pytest.mark.precommit
    @pytest.mark.parametrize("dict_type", ["int_int", "str_tensor"])
    def test_items_static_dict(self, dict_type, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dict_type),
            ie_device,
            precision,
            ir_version,
        )
