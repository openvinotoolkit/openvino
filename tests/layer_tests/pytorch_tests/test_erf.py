# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestErf(PytorchLayerTest):
    def _prepare_input(self, input_dtype, out=False):
        import numpy as np
        x = np.linspace(-3, 3).astype(input_dtype)
        if not out:
            return (x, )
        return (x, np.zeros_like(x).astype(input_dtype))

    def create_model(self, mode="", input_dtype="float32"):
        import torch
        dtypes = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32
        }

        dtype = dtypes[input_dtype]
        class aten_erf(torch.nn.Module):
            def __init__(self, mode, dtype):
                super(aten_erf, self).__init__()
                self.dtype = dtype
                if mode == "out":
                    self.forward = self.forward_out
                elif mode == "inplace":
                    self.forward = self.forward_inplace

            def forward(self, x):
                return torch.special.erf(x.to(self.dtype))

            def forward_out(self, x, y):
                return torch.special.erf(x.to(self.dtype), out=y), y

            def forward_inplace(self, x):
                x = x.to(self.dtype)
                return x.erf_(), x

        ref_net = None

        return aten_erf(mode, dtype), ref_net, "aten::erf" if mode != "inplace" else "aten::erf_"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_fx_backend
    @pytest.mark.parametrize("mode,input_dtype", [
        ("", "float32"), ("", "float64"), ("", "int32"),
        ("out", "float32"), ("out", "float64"),
        ("inplace", "float32"), ("inplace", "float64")])
    def test_erf(self, mode, input_dtype, ie_device, precision, ir_version):
        self._test(*self.create_model(mode, input_dtype), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"input_dtype": input_dtype, "out": mode == "out"} )