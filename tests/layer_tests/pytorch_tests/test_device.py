# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestDevice(PytorchLayerTest):
    def _prepare_input(self):
        input_data = self.random.randint(127, size=(1, 3, 224, 224))
        return (input_data,)

    def create_model_device(self, device_string):
        class prim_device(torch.nn.Module):
            def __init__(self, device_string) -> None:
                super().__init__()
                self.device_string = device_string

            def forward(self, x):
                if x.device == self.device_string:
                    return torch.add(x, 0)
                else:
                    return torch.add(x, 1)


        return prim_device(device_string), "prim::device"

    def create_model_type(self, device_string):
        import torch

        class prim_device(torch.nn.Module):
            def __init__(self, device_string) -> None:
                super().__init__()
                self.device_string = device_string

            def forward(self, x):
                if x.device.type == self.device_string:
                    return torch.add(x, 0)
                else:
                    return torch.add(x, 1)


        return prim_device(device_string), "prim::device"

    @pytest.mark.parametrize("device_string", ["cpu", "cuda"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_device(self, device_string, ie_device, precision, ir_version):
        self._test(
            *self.create_model_device(device_string),
            ie_device,
            precision,
            ir_version,
            trace_model=False,
            use_convert_model=True,
        )

    @pytest.mark.parametrize("device_string", ["cpu", "cuda"])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_device_type(self, device_string, ie_device, precision, ir_version):
        self._test(
            *self.create_model_type(device_string),
            ie_device,
            precision,
            ir_version,
            trace_model=False,
            use_convert_model=True,
        )
