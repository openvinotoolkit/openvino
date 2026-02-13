# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from pytorch_layer_test_class import PytorchLayerTest, skip_check, skip_if_export
import torch


class TestArange(PytorchLayerTest):
    def _prepare_input(self, start=None, end=None, step=None, ref_value=None, out=None):
        args = tuple(a for a in [start, end, step,
                     ref_value, out] if a is not None)
        return args

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("end", [np.array(1), np.array(2), np.array(3.)])
    def test_arange_end(self, ie_device, precision, ir_version, end):
        class aten_arange_end(torch.nn.Module):
            def forward(self, end):
                return torch.arange(end)

        self._test(aten_arange_end(), "aten::arange",
                   ie_device, precision, ir_version, trace_model=True,
                   kwargs_to_prepare_input={"end": end})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("end", [np.array(1), np.array(2), np.array(3.)])
    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64,
                                       torch.int8, torch.int32, torch.int64])
    def test_arange_end_dtype(self, ie_device, precision, ir_version, end, dtype):
        class aten_arange_end_dtype(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, end):
                return torch.arange(end, dtype=self.dtype)

        self._test(aten_arange_end_dtype(dtype), "aten::arange",
                   ie_device, precision, ir_version, trace_model=True,
                   kwargs_to_prepare_input={"end": end})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("end", [np.array(1), np.array(2), np.array(3.)])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64,
                                       torch.int8, torch.int32, torch.int64])
    def test_arange_end_prim_dtype(self, ie_device, precision, ir_version, end, dtype):
        class aten_arange_end_prim_dtype(torch.nn.Module):
            def forward(self, end, ref):
                return torch.arange(end, dtype=ref.dtype)

        self._test(aten_arange_end_prim_dtype(), ["aten::arange", "prim::dtype"],
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end,
                                            "ref_value": torch.tensor(1, dtype=dtype).numpy()})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("start,end", [(np.array(0), np.array(1)),
                                           (np.array(-1), np.array(1)),
                                           (np.array(0.4), np.array(2.4)),
                                           (np.array(16777215), np.array(16777247))])
    def test_arange_start_end(self, ie_device, precision, ir_version, start, end):
        class aten_arange_start_end(torch.nn.Module):
            def forward(self, start, end):
                return torch.arange(start, end)

        self._test(aten_arange_start_end(), "aten::arange",
                   ie_device, precision, ir_version, trace_model=True,
                   kwargs_to_prepare_input={"start": start, "end": end})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("start,end", [(np.array(0), np.array(1)),
                                           (np.array(-1), np.array(1)),
                                           (np.array(0.4), np.array(2.4)),
                                           (np.array(16777215), np.array(16777247))])
    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64,
                                       torch.int8, torch.int32, torch.int64])
    def test_arange_start_end_dtype(self, ie_device, precision, ir_version, start, end, dtype):
        if dtype == torch.int8 and start > 127:
            pytest.skip("int8 range is not supported in openvino")

        class aten_arange_start_end_dtype(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, start, end):
                return torch.arange(start, end, dtype=self.dtype)

        self._test(aten_arange_start_end_dtype(dtype), "aten::arange",
                   ie_device, precision, ir_version, trace_model=True,
                   kwargs_to_prepare_input={"start": start, "end": end})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("start,end", [
        (np.array(0), np.array(1)),
        (np.array(-1), np.array(1)),
        (np.array(0.4), np.array(2.4)),
        pytest.param(np.array(16777215), np.array(16777247), marks=pytest.mark.xfail(reason="unsupported for dynamic case"))])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64,
                                       torch.int8, torch.int32, torch.int64])
    def test_arange_start_end_prim_dtype(self, ie_device, precision, ir_version, start, end, dtype):
        if dtype == torch.int8 and start > 127:
            pytest.skip("int8 range is not supported in openvino")

        class aten_arange_start_end_prim_dtype(torch.nn.Module):
            def forward(self, start, end, ref):
                return torch.arange(start, end, dtype=ref.dtype)

        self._test(aten_arange_start_end_prim_dtype(), ["aten::arange", "prim::dtype"],
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"start": start, "end": end,
                                            "ref_value": torch.tensor(1, dtype=dtype).numpy()})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("start,end,step", [
        (np.array(0), np.array(1), np.array(1)),
        (np.array(-2), np.array(1), np.array(1.25)),
        (np.array(1), np.array(-5), np.array(-1)),
        (np.array(1), np.array(10), np.array(2)),
        (np.array(-1), np.array(-5), np.array(-2)),
        (np.array(16777215), np.array(16777247), np.array(1))])
    def test_arange_start_end_step(self, ie_device, precision, ir_version, start, end, step):
        class aten_arange_start_end_step(torch.nn.Module):
            def forward(self, start, end, step):
                return torch.arange(start, end, step)

        self._test(aten_arange_start_end_step(), "aten::arange",
                   ie_device, precision, ir_version, trace_model=True,
                   kwargs_to_prepare_input={"start": start, "end": end, "step": step})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("start,end,step", [
        (np.array(0), np.array(1), np.array(1)),
        (np.array(-2), np.array(1), np.array(1.25)),
        (np.array(1), np.array(-5), np.array(-1)),
        (np.array(1), np.array(10), np.array(2)),
        (np.array(-1), np.array(-5), np.array(-2)),
        (np.array(16777215), np.array(16777247), np.array(1))])
    @pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64,
                                       torch.int8, torch.int32, torch.int64])
    def test_arange_start_end_step_dtype(self, ie_device, precision, ir_version,
                                         start, end, step, dtype):
        if dtype == torch.int8 and start > 127:
            pytest.skip("int8 range is not supported in openvino")

        class aten_arange_start_end_step_dtype(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, start, end, step):
                return torch.arange(start, end, step, dtype=self.dtype)

        self._test(aten_arange_start_end_step_dtype(dtype), "aten::arange",
                   ie_device, precision, ir_version, trace_model=True,
                   kwargs_to_prepare_input={"start": start, "end": end, "step": step})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("start,end,step", [
        (np.array(0), np.array(1), np.array(1)),
        (np.array(-2), np.array(1), np.array(1.25)),
        (np.array(1), np.array(-5), np.array(-1)),
        (np.array(1), np.array(10), np.array(2)),
        (np.array(-1), np.array(-5), np.array(-2)),
        pytest.param(np.array(16777215), np.array(16777247), np.array(1), marks=pytest.mark.xfail(reason="unsupported for dynamic case"))])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64,
                                       torch.int8, torch.int32, torch.int64])
    def test_arange_start_end_step_prim_dtype(self, ie_device, precision, ir_version,
                                              start, end, step, dtype):
        class aten_arange_start_end_step_prim_dtype(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, start, end, step, ref):
                ref = ref.to(self.dtype)
                return torch.arange(start, end, step, dtype=ref.dtype)

        self._test(aten_arange_start_end_step_prim_dtype(dtype), ["aten::arange", "prim::dtype"],
                   ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"start": start, "end": end, "step": step,
                                            "ref_value": torch.tensor(1, dtype=dtype).numpy()})


class TestArangeLegacy(PytorchLayerTest):
    def _prepare_input(self, end, start=None, step=None, dtype="int64", ref_dtype=None):
        import numpy as np
        if start is None and step is None:
            return (np.array(end).astype(dtype),) if not ref_dtype else (np.array(end).astype(dtype), np.zeros(1).astype(ref_dtype))
        if step is None:
            return (np.array(start).astype(dtype), np.array(end).astype(dtype)) if not ref_dtype else (np.array(start).astype(dtype), np.array(end).astype(dtype), np.zeros(1).astype(ref_dtype))
        return (np.array(start).astype(dtype), np.array(end).astype(dtype), np.array(step).astype(dtype)) if not ref_dtype else (np.array(start).astype(dtype), np.array(end).astype(dtype), np.array(step).astype(dtype), np.zeros(1).astype(ref_dtype))

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

        class aten_arange_end_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.arange(x, dtype=self.dtype)

        class aten_arange_start_end_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.dtype = dtype

            def forward(self, x, y):
                return torch.arange(start=x, end=y, dtype=self.dtype)

        class aten_arange_start_end_step_dtype(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.dtype = dtype

            def forward(self, x, y, z):
                return torch.arange(start=x, end=y, step=z, dtype=self.dtype)

        class aten_arange_end_out(torch.nn.Module):
            def __init__(self, dtype) -> None:
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.arange(x, out=torch.zeros(1, dtype=self.dtype))

        class aten_arange_start_end_out(torch.nn.Module):
            def __init__(self, out) -> None:
                super().__init__()
                self.out = out

            def forward(self, x, y):
                return torch.arange(start=x, end=y, out=self.out)

        class aten_arange_start_end_step_out(torch.nn.Module):
            def __init__(self, out) -> None:
                super().__init__()
                self.out = out

            def forward(self, x, y, z):
                return torch.arange(start=x, end=y, step=z, out=self.out)

        class aten_arange_end_prim_dtype(torch.nn.Module):

            def forward(self, x, y):
                return torch.arange(x, dtype=y.dtype)

        class aten_arange_start_end_prim_dtype(torch.nn.Module):

            def forward(self, x, y, z):
                return torch.arange(start=x, end=y, dtype=z.dtype)

        class aten_arange_start_end_step_prim_dtype(torch.nn.Module):

            def forward(self, x, y, z, d):
                return torch.arange(start=x, end=y, step=z, dtype=d.dtype)

        model_classes = {
            1: (aten_arange_end_dtype, aten_arange_end_out, aten_arange_end_prim_dtype),
            2: (aten_arange_start_end_dtype, aten_arange_start_end_out, aten_arange_start_end_prim_dtype),
            3: (aten_arange_start_end_step_dtype, aten_arange_start_end_step_out, aten_arange_start_end_step_prim_dtype)
        }
        dtype = dtype_map.get(dtype)
        if ref_dtype:
            model_class = model_classes[num_inputs][2]()
        elif not use_out or dtype is None:
            model_class = model_classes[num_inputs][0](dtype)
        else:
            model_class = model_classes[num_inputs][1](dtype)


        return model_class, "aten::arange"

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dtype", [None,
                                       skip_if_export("float32"),
                                       skip_if_export("float64"),
                                       skip_if_export("int32"),
                                       "int64",
                                       skip_if_export("int8"),
                                       skip_if_export("uint8")])
    @pytest.mark.parametrize("end", [1, 2, 3])
    @pytest.mark.parametrize("use_out", [skip_check(True), False])
    def test_arange_end_only(self, dtype, end, use_out, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 1, use_out), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end}, trace_model=True)

    @pytest.mark.nightly
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize("start,end", [(0, 1), (-1, 1), (1, 5), (0.5, 2.5)])
    def test_arange_start_end(self, dtype, end, start, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 2), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end, "start": start, "dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", [None, "float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize("start,end,step", [(0, 1, 1), (-2, 1, 1.25), (1, -5, -1), (1, 10, 2), (-1, -5, -2)])
    def test_arange_start_end_step(self, dtype, end, start, step, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 3), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end, "start": start, "step": step, "dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dtype", [skip_check(None),
                                       skip_if_export("float32"),
                                       skip_if_export("float64"),
                                       skip_if_export("int32"),
                                       "int64",
                                       skip_if_export("int8"),
                                       skip_if_export("uint8")])
    @pytest.mark.parametrize("end", [1, 2, 3])
    def test_arange_end_only_with_prim_dtype(self, dtype, end, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 1, False, True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end, "ref_dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize("start,end", [(0, 1), (-1, 1), (1, 5), (0.5, 2.5)])
    def test_arange_start_end_with_prim_dtype(self, dtype, end, start, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 2, ref_dtype=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end, "start": start, "ref_dtype": dtype})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "int8"])
    @pytest.mark.parametrize("start,end,step", [(0, 1, 1), (-2, 1, 1.25), (1, -5, -1), (1, 10, 2), (-1, -5, -2)])
    def test_arange_start_end_step_with_prim_dtype(self, dtype, end, start, step, ie_device, precision, ir_version):
        self._test(*self.create_model(dtype, 3, ref_dtype=True), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"end": end, "start": start, "step": step, "ref_dtype": dtype})
