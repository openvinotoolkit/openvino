import numpy as np
import pytest
import torch
from common.layer_test_class import check_ie_class
from common.utils.pytorch_utils import get_torch_net_with_nodes


class VStackModel(torch.nn.Module):
    def forward(self, x1, x2):
        return torch.vstack((x1, x2))


class HStackModel(torch.nn.Module):
    def forward(self, x1, x2):
        return torch.hstack((x1, x2))


test_data_vstack = [
    # Test 1D tensors
    dict(
        input_data=[
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ],
        test_name="vstack_1d"
    ),
    # Test 2D tensors
    dict(
        input_data=[
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
        ],
        test_name="vstack_2d"
    ),
]

test_data_hstack = [
    # Test 1D tensors
    dict(
        input_data=[
            np.array([1, 2, 3], dtype=np.float32),
            np.array([4, 5, 6], dtype=np.float32),
        ],
        test_name="hstack_1d"
    ),
    # Test 2D tensors
    dict(
        input_data=[
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[5, 6], [7, 8]], dtype=np.float32),
        ],
        test_name="hstack_2d"
    ),
]


class TestVStack:
    @pytest.mark.parametrize("params", test_data_vstack)
    def test_vstack(self, params):
        model = VStackModel()
        x1, x2 = [torch.from_numpy(x) for x in params["input_data"]]
        node_name = "aten::vstack"
        model_ref = get_torch_net_with_nodes(model, (x1, x2), [node_name])
        check_ie_class(model=model_ref, input_data=params["input_data"])


class TestHStack:
    @pytest.mark.parametrize("params", test_data_hstack)
    def test_hstack(self, params):
        model = HStackModel()
        x1, x2 = [torch.from_numpy(x) for x in params["input_data"]]
        node_name = "aten::hstack"
        model_ref = get_torch_net_with_nodes(model, (x1, x2), [node_name])
        check_ie_class(model=model_ref, input_data=params["input_data"]) 