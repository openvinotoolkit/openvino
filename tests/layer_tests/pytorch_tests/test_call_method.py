
import pytest
import torch
import numpy as np
from pytorch_layer_test_class import PytorchLayerTest

class TestCallMethod(PytorchLayerTest):

    def _prepare_input(self):
        return (np.random.randn(1, 3, 224, 224).astype(np.float32),)

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
        def __init__(self):
            super().__init__()

        def complex_method(self, x, y, scale):
            return (x + y) * scale

        def forward(self, x):
            return self.complex_method(x, x, 0.5)

    def test_call_method(self, model_class, ie_device, precision, ir_version):
        model = model_class()
        # skip_freeze=True ensures prim::CallMethod is preserved in the graph 
        # and not automatically inlined by the JIT before our frontend sees it,
        # verifying our explicit translator.
        
        # Test 1: Scripted model with skip_freeze=True (keeps prim::CallMethod)
        scripted = torch.jit.script(model)
        self._test(scripted, None, "prim::CallMethod", 
                   ie_device, precision, ir_version, 
                   trace_model=False,
                   freeze_model=False) # freeze_model=False -> passes skip_freeze=True to decoder

    def test_call_method_traced(self, model_class, ie_device, precision, ir_version):
        # Tracing usually inlines, but we test it anyway to ensure no regressions
        model = model_class()
        self._test(model, None, "aten::add", # Expect inlined ops
                   ie_device, precision, ir_version, 
                   trace_model=True, 
                   freeze_model=True) 
