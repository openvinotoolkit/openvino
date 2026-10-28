# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from pytorch_layer_test_class import PytorchLayerTest


class TestDiff(PytorchLayerTest):
    def _prepare_input(self, dtype):
        if dtype == "bool":
            return (self.random.randint(0, 2, (3, 4, 5)).astype(bool),)
        return (self.random.randn(3, 4, 5).astype(dtype),)

    def create_model(self, n, dim, prepend, append):
        class aten_diff(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.n, self.dim, self.prepend, self.append = n, dim, prepend, append

            def forward(self, x):
                pre = x.narrow(self.dim, 0, 1) if self.prepend else None
                app = x.narrow(self.dim, 0, 2) if self.append else None
                return torch.diff(x, self.n, self.dim, pre, app)

        return aten_diff(), "aten::diff"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("n", [1, 2])
    @pytest.mark.parametrize("dim", [-1, 0, 1])
    @pytest.mark.parametrize("prepend", [False, True])
    @pytest.mark.parametrize("append", [False, True])
    @pytest.mark.parametrize("dtype", ["float32", "int64"])
    def test_diff(self, n, dim, prepend, append, dtype, ie_device, precision, ir_version):
        if ie_device == "GPU" and (prepend or append):
            pytest.xfail("GPU plugin miscompiles Slice(Concat(a, x), 0:-1) - Subtract shape mismatch")
        self._test(*self.create_model(n, dim, prepend, append), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_diff_bool(self, ie_device, precision, ir_version):
        # Scripting leaves the input type dynamic, tracing exposes bool to the converter.
        self._test(*self.create_model(1, -1, False, False), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"dtype": "bool"}, trace_model=True)
