# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from pytorch_layer_test_class import PytorchLayerTest


class TestLinspace(PytorchLayerTest):
    def _prepare_input(self, end, start=None, steps=None, dtype="int64", ref_dtype=None):
        import numpy as np

        if start is None and steps is None:
            return (
                (np.array(end).astype(dtype),)
                if not ref_dtype
                else (np.array(end).astype(dtype), np.zeros(1).astype(ref_dtype))
            )
        if steps is None:
            return (
                (np.array(start).astype(dtype), np.array(end).astype(dtype))
                if not ref_dtype
                else (np.array(start).astype(dtype), np.array(end).astype(dtype), np.zeros(1).astype(ref_dtype))
            )
        return (
            (np.array(start).astype(dtype), np.array(end).astype(dtype), np.array(steps).astype(dtype))
            if not ref_dtype
            else (
                np.array(start).astype(dtype),
                np.array(end).astype(dtype),
                np.array(steps).astype("int32"),
                np.zeros(1).astype(ref_dtype),
            )
        )

    def create_model(self, dtype=None, num_inputs=1, use_out=False, ref_dtype=False):
        import torch

        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
        }

        class aten_linspace_start_end_steps_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super(aten_linspace_start_end_steps_dtype, self).__init__()
                self.dtype = dtype

            def forward(self, x, y, z):
                return torch.linspace(start=x, end=y, steps=z, dtype=self.dtype)

        class aten_linspace_start_end_steps_out(torch.nn.Module):
            def __init__(self, out) -> None:
                super(aten_linspace_start_end_steps_out, self).__init__()
                # Size of empty tensor needs to be of equal or larger size than linspace steps
                self.out = torch.empty(25, dtype=out)

            def forward(self, x, y, z):
                return torch.linspace(start=x, end=y, steps=z, out=self.out)

        class aten_linspace_start_end_steps_prim_dtype(torch.nn.Module):
            def forward(self, x, y, z, d):
                return torch.linspace(start=x, end=y, steps=z, dtype=d.dtype)

        model_classes = {
            3: (
                aten_linspace_start_end_steps_dtype,
                aten_linspace_start_end_steps_out,
                aten_linspace_start_end_steps_prim_dtype,
            )
        }
        dtype = dtype_map.get(dtype)
        if ref_dtype:
            model_class = model_classes[num_inputs][2]()
        elif not use_out or dtype is None:
            model_class = model_classes[num_inputs][0](dtype)
        else:
            model_class = model_classes[num_inputs][1](dtype)

        ref_net = None

        return model_class, ref_net, "aten::linspace"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "dtype",
        [
            # None,
            # "float32",
            # "float64",
            "int32",
            "int64",
            "int8",
        ],
    )
    @pytest.mark.parametrize("start,end,steps", [(0, 1, 1), (5, 7, 1), (-2, 1, 5), (1, -5, 7), (1, 10, 2), (-1, -5, 2)])
    def test_linspace_start_end_steps(self, dtype, end, start, steps, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype, 3),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"end": end, "start": start, "steps": steps, "dtype": dtype}
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "dtype",
        [
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
        ],
    )
    @pytest.mark.parametrize(
        "start,end,steps", [(0, 1, 5), (-2, 1, 5), (1, -5, 7), (1, 10, 2), (-1, -5, 2), (-1, -5, 1)]
    )
    def test_linspace_start_end_steps_with_prim_dtype(self, dtype, end, start, steps, ie_device, precision, ir_version):
        self._test(
            *self.create_model(dtype, 3, ref_dtype=True),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"end": end, "start": start, "steps": steps, "ref_dtype": dtype}
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize(
        "dtype",
        [
            # None,
            "float32",
            "float64",
            "int32",
            "int64",
            "int8",
            "uin8",
        ],
    )
    @pytest.mark.parametrize(
        "start,end,steps", [(0, 1, 5), (-2, 1, 5), (1, -5, 7), (1, 10, 2), (-1, -5, 2), (-1, -5, 1)]
    )
    @pytest.mark.parametrize("use_out", [False, True])
    def test_linspace_start_end_steps_with_out(
        self, dtype, use_out, end, start, steps, ie_device, precision, ir_version
    ):
        self._test(
            *self.create_model(dtype=dtype, num_inputs=3, use_out=use_out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={"end": end, "start": start, "steps": steps}
        )
