# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestExp(PytorchLayerTest):
    def _prepare_input(self, end, start=None, step=None, dtype="int64"):
        import numpy as np
        if start is None and step is None:
            return (np.array(end).astype(dtype),)
        if step is None:
            return (np.array(start).astype(dtype), np.array(end).astype(dtype))
        return (np.array(start).astype(dtype), np.array(end).astype(dtype), np.array(step).astype(dtype))

    def create_model(self, dtype, num_inputs, use_out=False):
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8
        }

        class aten_arange_end_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super(aten_arange_end_dtype, self).__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.arange(x, dtype=self.dtype)

        class aten_arange_start_end_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super(aten_arange_start_end_dtype, self).__init__()
                self.dtype = dtype

            def forward(self, x, y):
                return torch.arange(start=x, end=y, dtype=self.dtype)

        class aten_arange_start_end_step_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super(aten_arange_start_end_step_dtype, self).__init__()
                self.dtype = dtype

            def forward(self, x, y, z):
                return torch.arange(start=x, end=y, step=z, dtype=self.dtype)

        class aten_arange_end_out(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super(aten_arange_end_out, self).__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.arange(x, out=torch.zeros(1, dtype=self.dtype))

        class aten_arange_start_end_out(torch.nn.Module):
            def __init__(self, out) -> None:
                super(aten_arange_start_end_out, self).__init__()
                self.out = out

            def forward(self, x, y):
                return torch.arange(start=x, end=y, out=self.out)

        class aten_arange_start_end_step_out(torch.nn.Module):
            def __init__(self, out) -> None:
                super(aten_arange_start_end_step_out, self).__init__()
                self.out = out

            def forward(self, x, y, z):
                return torch.arange(start=x, end=y, step=z, out=self.out)

        model_classes = {
            1: (aten_arange_end_dtype, aten_arange_end_out),
            2: (aten_arange_start_end_dtype, aten_arange_start_end_out),
            3: (aten_arange_start_end_step_dtype, aten_arange_start_end_step_out)
        }
        dtype = dtype_map.get(dtype)
        model_class = model_classes[num_inputs][0](dtype) if not use_out or dtype is None else \
        model_classes[num_inputs][1](dtype)
        print(model_class)

        ref_net = None

        return model_class, ref_net, "aten::arange"

    @pytest.mark.nightly
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8", "uin8"])
    @pytest.mark.parametrize("end", [1, 2, 3])
    @pytest.mark.parametrize("use_out", [True, False])
    def test_arange_end_only(self, dtype, end, use_out, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 1, use_out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end})

    @pytest.mark.nightly
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize("start,end", [(0, 1), (-1, 1), (1, 5), (0.5, 2.5)])
    def test_arange_start_end(self, dtype, end, start, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 2), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end, "start": start, "dtype": "float32"})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize("start,end,step", [(0, 1, 1), (-2, 1, 1.25), (1, -5, -1), (1, 10, 2), (-1, -5, -2)])
    def test_arange_start_end_step(self, dtype, end, start, step, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 3), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end, "start": start, "step": step, "dtype": "float32"})
