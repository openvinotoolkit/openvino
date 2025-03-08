# test_aten_contains.py

import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestContains(PytorchLayerTest):
    def _prepare_input(self, container, element):
        """Prepare inputs for __contains__ testing"""
        container = torch.tensor(container)
        return container, element

    def create_model(self):
        class ContainsModel(torch.nn.Module):
            def forward(self, container, element):
                return element in container

        model_class = ContainsModel()
        ref_net = None

        return model_class, ref_net, "aten::__contains__"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "container, element",
        [
            ([1, 2, 3, 4], 3), 
            ([1, 2, 3, 4], 5), 
            ([], 1),           
            ([1], 1),          
            ([1, 2, 3, 3, 4], 3), 
            ([-1, -2, -3], -2),    
            ([0, 1, 2], 0),        
            ([True, False], True), 
        ],
    )
    def test_contains_basic(self, container, element, ie_device, precision, ir_version):
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"container": container, "element": element},
            expected_result=(element in container)
        )

    @pytest.mark.parametrize(
        "container_type",
        [
            list,               
            torch.tensor,       
            set,                
            tuple,              
        ],
    )
    def test_contains_container_types(self, container_type, ie_device, precision, ir_version):
        container = container_type([1, 2, 3, 4])
        element = 2
        expected_result = element in container

        if container_type == torch.tensor:
            container = torch.tensor([1, 2, 3, 4])

        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"container": container, "element": element},
            expected_result=expected_result
        )

    @pytest.mark.parametrize(
        "dtype",
        [
            torch.int32,
            torch.int64,
            torch.float32,
            torch.float64,
        ],
    )
    def test_contains_with_dtypes(self, dtype, ie_device, precision, ir_version):
        container = torch.tensor([1, 2, 3, 4], dtype=dtype)
        element = 3 if dtype.is_floating_point else 3
        expected_result = element in container

        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"container": container.tolist(), "element": element},
            expected_result=expected_result
        )

    @pytest.mark.parametrize(
        "element_type",
        [
            int,      
            float,   
            bool,     
        ],
    )
    def test_contains_element_types(self, element_type, ie_device, precision, ir_version):
        container = torch.tensor([1, 2, 3, 4])
        element = element_type(3)
        expected_result = element in container

        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"container": container.tolist(), "element": element},
            expected_result=expected_result
        )

    def test_contains_large_container(self, ie_device, precision, ir_version):
        container = torch.arange(0, 1000000)
        element = 999999  
        expected_result = element in container

        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"container": container.tolist(), "element": element},
            expected_result=expected_result
        )
