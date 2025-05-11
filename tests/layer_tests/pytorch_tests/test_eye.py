# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from packaging import version

from pytorch_layer_test_class import PytorchLayerTest


class TestEye(PytorchLayerTest):
    def _prepare_input(self, m, n=None):
        import numpy as np
        if n is None:
            return (np.array(m, dtype="int32"), )
        return (np.array(m, dtype="int32"), np.array(n, dtype="int32"))

    def create_model(self, num_inputs, dtype):
        import torch
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int64": torch.int64,
            "int32": torch.int32,
            "uint8": torch.uint8,
            "int8": torch.int8,
            "bool": torch.bool
        }

        pt_dtype = dtype_map.get(dtype)

        class aten_eye_1_input(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return torch.eye(x, dtype=self.dtype)

        class aten_eye_2_inputs(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x, y):
                return torch.eye(x, y, dtype=self.dtype)

        model = aten_eye_1_input(pt_dtype) if num_inputs == 1 else aten_eye_2_inputs(pt_dtype)
        return model, None, ["aten::eye", "aten::IntImplicit"]

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize("m", [2, 3, 4, 5])
    def test_eye_square(self, dtype, m, ie_device, precision, ir_version):
        if PytorchLayerTest.use_torch_export() and version.parse(torch.__version__) < version.parse("2.3"):
            pytest.skip("Not supported in PyTorch versions earlier than 2.3.")
        if ie_device == "GPU":
            pytest.xfail(reason="eye is not supported on GPU")
        self._test(*self.create_model(1, dtype), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"m": m})

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dtype", ["bool", "int8", "uint8", "int32", "int64", "float32", "float64"])
    @pytest.mark.parametrize(("m", "n"), [[2, 2], [3, 4], [5, 3]])
    def test_eye(self, dtype, m, n, ie_device, precision, ir_version):
        if (PytorchLayerTest.use_torch_export() and version.parse(torch.__version__) < version.parse("2.3")):
            pytest.skip("Not supported in PyTorch versions earlier than 2.3.")
        if ie_device == "GPU":
            pytest.xfail(reason="eye is not supported on GPU")
        self._test(*self.create_model(2, dtype), ie_device, precision,
                   ir_version, kwargs_to_prepare_input={"m": m, "n": n})
