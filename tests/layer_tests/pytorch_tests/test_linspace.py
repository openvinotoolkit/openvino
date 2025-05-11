# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestLinspace(PytorchLayerTest):
    def _prepare_input(self, start, end, steps, dtype=None, ref_dtype=None):
        inputs = [np.array(start).astype(dtype), np.array(end).astype(dtype), np.array(steps).astype("int32")]
        if ref_dtype:
            inputs.append(np.zeros(1).astype(ref_dtype))
        return inputs

    def create_model(self, dtype=None, use_out=False, ref_dtype=False):
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
        }

        class aten_linspace_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.dtype = dtype

            def forward(self, start, end, steps):
                return torch.linspace(start=start, end=end, steps=steps, dtype=self.dtype)

        class aten_linspace_out(torch.nn.Module):
            def __init__(self, out) -> None:
                super().__init__()
                # Size of empty tensor needs to be of equal or larger size than linspace steps
                self.out = torch.empty(25, dtype=out)

            def forward(self, start, end, steps):
                return torch.linspace(start=start, end=end, steps=steps, out=self.out)

        class aten_linspace_prim_dtype(torch.nn.Module):
            def forward(self, start, end, steps, d):
                return torch.linspace(start=start, end=end, steps=steps, dtype=d.dtype)

        dtype = dtype_map.get(dtype)
        if ref_dtype:
            model_class = aten_linspace_prim_dtype()
        elif not use_out:
            model_class = aten_linspace_dtype(dtype)
        else:
            model_class = aten_linspace_out(dtype)

        ref_net = None

        return model_class, ref_net, "aten::linspace"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize(
        "start,end,steps", [(0, 1, 5), (-2, 1, 5), (1, -5, 7), (1, 10, 2), (-1, -5, 2), (-1, -5, 1), (1.25, -5.5, 5)]
    )
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_linspace_with_prim_dtype(self, dtype, end, start, steps, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype, ref_dtype=True),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"end": end, "start": start, "steps": steps, "ref_dtype": dtype}
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8", "uin8"])
    @pytest.mark.parametrize(
        "start,end,steps", [(0, 1, 5), (-2, 1, 5), (1, -5, 7), (1, 10, 2), (-1, -5, 2), (-1, -5, 1), (1.25, -5.5, 5)]
    )
    @pytest.mark.parametrize("use_out", [False, True])
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122715')
    def test_linspace_with_out(self, dtype, use_out, end, start, steps, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype=dtype, use_out=use_out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"end": end, "start": start, "steps": steps}
        )
