# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest, skip_if_export


class TestCeil(PytorchLayerTest):
    def _prepare_input(self, out=False, dtype="float32"):
        import numpy as np
        input = self.random.randn(1, 3, 224, 224, dtype=dtype)

        if dtype == "float64":
            input = np.round(input, 6)

        if not out:
            return (input, )
        return (input, np.zeros_like(input))

    def create_model(self, out=False):
        import torch

        class aten_ceil(torch.nn.Module):
            def __init__(self, out):
                super().__init__()
                if out:
                    self.forward = self.forward_out

            def forward(self, x):
                return torch.ceil(x)

            def forward_out(self, x, y):
                return torch.ceil(x, out=y), y

        return aten_ceil(out), "aten::ceil"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("out", [skip_if_export(True), False])
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
    def test_ceil(self, out, dtype, ie_device, precision, ir_version):
        if ie_device == "GPU" and dtype not in ["float32", "float64"]:
            pytest.xfail(reason="ceil is not supported on GPU for integer types")

        self._test(
            *self.create_model(out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"out": out, "dtype": dtype}
        )