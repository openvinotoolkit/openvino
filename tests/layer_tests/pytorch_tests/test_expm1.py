# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0i

import pytest

from pytorch_layer_test_class import PytorchLayerTest

class TestExpm1(PytorchLayerTest):
    def _prepare_input(self, inputs, dtype=None, out=False):
        import numpy as np
        x = np.array(inputs).astype(dtype)
        if not out:
            return (x, )
        return (x, np.zeros_like(x).astype(dtype))

    def create_model(self, mode="", dtype=None):
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }

        dtype = dtype_map.get(dtype)

        class aten_expm1(torch.nn.Module):
            def __init__(self, mode, dtype):
                super().__init__()
                self.dtype = dtype
                if mode == "out":
                    self.forward = self.forward_out
                else:
                    self.forward = self.forward_default

            def forward_default(self, x):
                return torch.expm1(x.to(self.dtype)).to(torch.float32)

            def forward_out(self, x, y):
                y = y.to(torch.float32)
                return torch.expm1(x.to(self.dtype), out=y).to(torch.float32), y

        model_class = aten_expm1(mode, dtype)

        ref_net = None

        return model_class, ref_net, "aten::expm1"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("mode,dtype", [
        ("", "float32"), ("", "float64"), ("", "int32"), ("", "int64"),
        ("out", "float32"), ("out", "float64"), ("out", "int32"), ("out", "int64")])
    @pytest.mark.parametrize("inputs", [[0, 1, 2, 3, 4, 5], [-2, -1, 0, 1, 2, 3], [1, 2, 3, 4, 5, 6]])
    def test_expm1(self, mode, dtype, inputs, ie_device, precision, ir_version):
        self._test(
            *self.create_model(mode, dtype),
            ie_device,
            precision,
            ir_version,
            trace_model=True,
            freeze_model=False,
            kwargs_to_prepare_input={"inputs": inputs, "dtype": dtype, "out": mode == "out"}        
        )
