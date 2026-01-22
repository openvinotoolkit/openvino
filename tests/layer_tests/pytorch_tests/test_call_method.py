# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest


class TestCallMethod(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

    def create_model(self, model_cls):
        import torch

        class SimpleMethodClass(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.val = 1.0

            def custom_method(self, x):
                return x.add(self.val)

            def forward(self, x):
                return self.custom_method(x)

        class NestedMethodClass(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.multiplier = 2.0

            def inner_method(self, x):
                return x * self.multiplier

            def outer_method(self, x):
                return self.inner_method(x) + 1.0

            def forward(self, x):
                return self.outer_method(x)

        class MultiArgMethodClass(torch.nn.Module):
            def complex_method(self, x, y, scale):
                return (x + y) * scale

            def forward(self, x):
                return self.complex_method(x, x, 0.5)

        models = {
            "simple": SimpleMethodClass,
            "nested": NestedMethodClass,
            "multi_arg": MultiArgMethodClass,
        }
        model = torch.jit.script(models[model_cls]())
        return model, "prim::CallMethod"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("model_cls", ["simple", "nested", "multi_arg"])
    def test_call_method(self, model_cls, ie_device, precision, ir_version):
        self._test(*self.create_model(model_cls),
                   ie_device, precision, ir_version,
                   trace_model=False, freeze_model=False)
