# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
from pytorch_layer_test_class import PytorchLayerTest


class TestSlice1D(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array(range(16), np.float32), np.array(self.params, dtype=np.int32))

    def create_model(self):
        class aten_slice(torch.nn.Module):
            def forward(self, x, params):
                return x[params[0] : params[1] : params[2]]


        return aten_slice(), "aten::slice"

    @pytest.mark.parametrize(
        "params",
        [[0, -1, 1], [0, -1, 3], [0, 5, 3], [2, 7, 3], [-7, -15, 2], [-1, -7, 2], [5, 2, 1]],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_slice1d(self, ie_device, precision, ir_version, params):
        self.params = params
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
        )


class TestSlice2D(PytorchLayerTest):
    def _prepare_input(self):
        return (
            np.array([range(16), range(16, 32)], np.float32),
            np.array(self.params_0a, dtype=np.int32),
            np.array(self.params_1a, dtype=np.int32),
        )

    def create_model(self):
        class aten_slice(torch.nn.Module):
            def forward(self, x, params_0a, params_1a):
                return x[params_0a[0] : params_0a[1] : params_0a[2], params_1a[0] : params_1a[1] : params_1a[2]]


        return aten_slice(), "aten::slice"

    @pytest.mark.parametrize(
        "params_0a",
        [[0, -1, 1], [0, -1, 3], [0, 5, 3], [2, 7, 3], [-7, -15, 2], [-1, -7, 2], [5, 2, 1]],
    )
    @pytest.mark.parametrize(
        "params_1a",
        [[0, -1, 1], [0, -1, 3], [0, 5, 3], [2, 7, 3], [-7, -15, 2], [-1, -7, 2], [5, 2, 1]],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_slice2d(self, ie_device, precision, ir_version, params_0a, params_1a):
        self.params_0a = params_0a
        self.params_1a = params_1a
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
        )


class TestSliceComplex(PytorchLayerTest):
    def _prepare_input(self, params):
        return (np.array(range(32), np.float32).reshape(16, 2),
                np.array(params, dtype=np.int32))

    def create_model(self):
        class aten_slice(torch.nn.Module):
            def forward(self, x, params):
                x = torch.view_as_complex(x)
                x = x[params[0]: params[1]: params[2]]
                return torch.view_as_real(x)

        return aten_slice(), "aten::slice"

    @pytest.mark.parametrize("params", [[0, -1, 1],
                                        [0, -1, 3],
                                        [0, 5, 3],
                                        [2, 7, 3],
                                        [-7, -15, 2],
                                        [-1, -7, 2],
                                        [5, 2, 1]])
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_slice_complex(self, ie_device, precision, ir_version, params):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   kwargs_to_prepare_input={"params": params})


class TestSliceAndSqueeze(PytorchLayerTest):
    def _prepare_input(self):
        return (self.random.randn(1, 1, 32),)

    def create_model(self):
        class aten_slice(torch.nn.Module):
            def forward(self, x):
                a = torch.squeeze(x, 1)
                return a[:, None, :]

        return aten_slice(), "aten::slice"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    def test_slice_and_squeeze(self, ie_device, precision, ir_version):
        self._test(*self.create_model(), ie_device, precision, ir_version,
                   dynamic_shapes=False, fx_kind="aten.unsqueeze.default")


class TestSliceListNegativeStep(PytorchLayerTest):
    """aten::slice.t with a negative step, where omitted bounds must default to the far ends.

    Tensors reject negative steps in PyTorch, but TorchScript list slicing (e.g. `x.shape[::-1]`)
    allows them, and an omitted start/end then means "last element" / "past the first element".
    """

    def _prepare_input(self):
        return (self.random.randn(2, 3, 4).astype(np.float32), np.array(-1, dtype=np.int32))

    def create_model(self, case):
        class aten_slice_reverse(torch.nn.Module):
            def forward(self, x, step):
                return torch.ones(x.shape[::-1]) * x.sum()

        class aten_slice_reverse_from_start(torch.nn.Module):
            def forward(self, x, step):
                return torch.ones(x.shape[1::-1]) * x.sum()

        class aten_slice_reverse_to_end(torch.nn.Module):
            def forward(self, x, step):
                return torch.ones(x.shape[:0:-1]) * x.sum()

        class aten_slice_reverse_step2(torch.nn.Module):
            def forward(self, x, step):
                return torch.ones(x.shape[::-2]) * x.sum()

        class aten_slice_forward_step2(torch.nn.Module):
            def forward(self, x, step):
                return torch.ones(x.shape[::2]) * x.sum()

        class aten_slice_reverse_dynamic_step(torch.nn.Module):
            # The step is only known at runtime, so the bounds are selected in the graph. The sliced
            # list then has a dynamic length, which torch.ones could not consume, so return it as is.
            def forward(self, x, step):
                return torch.tensor(x.shape[::int(step)]) + x.sum().to(torch.int64)

        models = {
            "reverse": aten_slice_reverse,
            "reverse_from_start": aten_slice_reverse_from_start,
            "reverse_to_end": aten_slice_reverse_to_end,
            "reverse_step2": aten_slice_reverse_step2,
            "forward_step2": aten_slice_forward_step2,
            "reverse_dynamic_step": aten_slice_reverse_dynamic_step,
        }
        return models[case](), "aten::slice"

    @pytest.mark.parametrize(
        "case",
        ["reverse", "reverse_from_start", "reverse_to_end", "reverse_step2", "forward_step2", "reverse_dynamic_step"],
    )
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_slice_list_negative_step(self, ie_device, precision, ir_version, case):
        if ie_device == "GPU" and case == "reverse_dynamic_step":
            pytest.xfail(reason="GPU skips a runtime-strided slice as identity, see openvinotoolkit/openvino#38240")
        self._test(*self.create_model(case), ie_device, precision, ir_version)
