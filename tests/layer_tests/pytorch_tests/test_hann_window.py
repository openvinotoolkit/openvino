# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestHannWindow(PytorchLayerTest):
    def _prepare_input(self, window_size, out=False, out_dtype="float32"):
        import numpy as np

        if not out:
            return (np.array(window_size),)
        return (np.array(window_size), np.zeros((window_size,), dtype=out_dtype))

    def create_model(self, periodic, dtype, out):
        import torch

        dtype_mapping = {
            "float32": torch.float32,
            "float64": torch.float64,
            "float16": torch.float16
        }

        torch_dtype = dtype_mapping.get(dtype)

        class aten_hann_window(torch.nn.Module):
            def __init__(self, periodic, dtype, out):
                super(aten_hann_window, self).__init__()
                self.periodic = periodic
                self.dtype = dtype

                if out:
                    self.forward = self.forward_out if periodic is None else self.forward_periodic_out
                elif dtype:
                    self.forward = self.forward_dtype if periodic is None else self.forward_dtype_periodic
                elif periodic is not None:
                    self.forward = self.forward_periodic

            def forward(self, x):
                return torch.hann_window(x)

            def forward_out(self, x, out):
                return torch.hann_window(x, out=out)

            def forward_periodic_out(self, x, out):
                return torch.hann_window(x, periodic=self.periodic, out=out)

            def forward_dtype(self, x):
                return torch.hann_window(x, dtype=self.dtype)

            def forward_dtype_periodic(self, x):
                return torch.hann_window(x, periodic=self.periodic, dtype=self.dtype)

            def forward_periodic(self, x):
                return torch.hann_window(x, periodic=self.periodic)

        ref_net = None

        return aten_hann_window(periodic, torch_dtype, out), ref_net, "aten::hann_window"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("window_size", [2, 10, 32])
    @pytest.mark.parametrize(("dtype", "out", "out_dtype", "periodic"), [
        [None, False, None, None], 
        [None, True, "float32", None],
        [None, True, "float64", None],
        [None, True, "float32", False],  
        [None, True, "float64", False], 
        [None, True, "float32", True],
        [None, True, "float64", True],
        [None, False, "", False],
        [None, False, "", True],
        ["float32", False, "", None],
        ["float64", False, "", None],
        ["float32", False, "", False],
        ["float64", False, "", False],
        ["float32", False, "", True],
        ["float64", False, "", True],
        ])
    def test_hann_window(self, window_size, dtype, out, out_dtype,  periodic, ie_device, precision, ir_version):
        self._test(*self.create_model(periodic, dtype, out), ie_device, precision, 
                   ir_version, kwargs_to_prepare_input={"window_size": window_size, "out": out, "out_dtype": out_dtype})