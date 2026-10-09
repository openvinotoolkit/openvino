# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestCopy(PytorchLayerTest):
    def _prepare_input(self):
        import numpy as np
        return (self.random.randn(1, 3, 224, 224),)

    def create_model(self, value):
        import torch

        class aten_copy(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.value = torch.tensor(value)

            def forward(self, x):
                return x.copy_(self.value)


        return aten_copy(value), "aten::copy_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("value", [1, [2.5], range(224)])
    def test_copy_(self, value, ie_device, precision, ir_version):
        self._test(*self.create_model(value), ie_device, precision, ir_version)


@pytest.mark.skipif(
    not (PytorchLayerTest.use_torch_export() or PytorchLayerTest.use_torch_compile_backend()),
    reason="Tests the functional ATen copy used by FX graphs",
)
class TestFunctionalCopy(PytorchLayerTest):
    def _prepare_input(self, source_shape):
        return (self.random.randn(2, 3).astype("float32"),
                self.random.randint(-10, 10, size=source_shape).astype("int32"))

    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.nightly
    @pytest.mark.parametrize("non_blocking", [None, False, True])
    @pytest.mark.parametrize("source_shape", [(2, 3), (3,)])
    def test_copy(self, non_blocking, source_shape, ie_device, precision, ir_version):
        import torch

        class Model(torch.nn.Module):
            def forward(self, destination, source):
                if non_blocking is None:
                    copied = torch.ops.aten.copy.default(destination, source)
                else:
                    copied = torch.ops.aten.copy.default(destination, source, non_blocking)
                return copied, destination, source

        self._test(Model(), "aten::copy", ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"source_shape": source_shape})


class TestAliasCopy(PytorchLayerTest):
    def _prepare_input(self, out):
        import numpy as np
        if not out:
            return (self.random.randn(1, 3, 224, 224),)
        return (self.random.randn(1, 3, 224, 224), np.zeros((1, 3, 224, 224), dtype=np.float32))

    def create_model(self, out):
        import torch

        class aten_copy(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                return torch.alias_copy(x)

            def forward_out(self, x, y):
                return torch.alias_copy(x, out=y), y


        return aten_copy(out), "aten::alias_copy"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("out", [True, False])
    def test_copy_(self, out, ie_device, precision, ir_version):
        self._test(*self.create_model(out), ie_device, precision, ir_version, kwargs_to_prepare_input={"out": out})
