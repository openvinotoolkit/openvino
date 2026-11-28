# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from pytorch_layer_test_class import PytorchLayerTest


class TestLogAddExp2(PytorchLayerTest):
    def _prepare_input(self, input1, input2, dtype1="float32", dtype2="float32", out=False):
        inp1 = np.asarray(input1, dtype=dtype1)
        inp2 = np.asarray(input2, dtype=dtype2)
        if not out:
            return (inp1, inp2)
        return (inp1, inp2, np.zeros_like(np.logaddexp2(inp1, inp2)))

    def create_model(self, out=False):
        import torch

        class LogAddExp2Model(torch.nn.Module):
            def __init__(self, use_out):
                super().__init__()
                if use_out:
                    self.forward = self.forward_out

            def forward(self, x, y):
                return torch.logaddexp2(x, y)

            def forward_out(self, x, y, output):
                return torch.logaddexp2(x, y, out=output), output

        return LogAddExp2Model(out), "aten::logaddexp2"

    @pytest.mark.parametrize(
        "dtype1,dtype2",
        [
            ("float16", "float16"),
            ("float32", "float32"),
            ("float64", "float64"),
            ("float32", "float64"),
            ("float16", "float32"),
        ],
    )
    @pytest.mark.parametrize(
        "input1,input2",
        [
            (0.0, 0.0),
            (1.0, 1.0),
            (-1000.0, -1000.0),
            (1024.0, 1024.0),
            (1000.0, -1000.0),
            (np.nan, 1.0),
        ],
    )
    @pytest.mark.parametrize("out", [False, True])
    def test_logaddexp2_values(self, dtype1, dtype2, input1, input2, out, ie_device, precision, ir_version):
        tolerance = 1e-3 if "float16" in (dtype1, dtype2) else 1e-5
        self._test(
            *self.create_model(out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "input1": input1,
                "input2": input2,
                "dtype1": dtype1,
                "dtype2": dtype2,
                "out": out,
            },
            rtol=tolerance,
            atol=tolerance,
        )

    @pytest.mark.parametrize("dtype", ["float16", "float32"])
    @pytest.mark.parametrize(
        "input1,input2",
        [
            (np.inf, 0.0),
            (0.0, np.inf),
            (-np.inf, 3.0),
            (3.0, -np.inf),
            (-np.inf, -np.inf),
            (np.inf, np.inf),
            (np.inf, -np.inf),
        ],
    )
    @pytest.mark.parametrize("out", [False, True])
    def test_logaddexp2_infinities(self, dtype, input1, input2, out, ie_device, precision, ir_version):
        tolerance = 1e-3 if dtype == "float16" else 1e-5
        self._test(
            *self.create_model(out),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "input1": input1,
                "input2": input2,
                "dtype1": dtype,
                "dtype2": dtype,
                "out": out,
            },
            rtol=tolerance,
            atol=tolerance,
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dtype1,dtype2", [("float16", "float16"), ("float32", "float32"), ("float32", "float64")])
    @pytest.mark.parametrize("shape1,shape2", [((3,), (3,)), ((2, 1), (1, 3)), ((1, 2, 1), (3,))])
    def test_logaddexp2_shapes(self, dtype1, dtype2, shape1, shape2, ie_device, precision, ir_version):
        tolerance = 1e-3 if "float16" in (dtype1, dtype2) else 1e-5
        input1 = self.random.uniform(-100, 100, shape1).astype(dtype1)
        input2 = self.random.uniform(-100, 100, shape2).astype(dtype2)
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "input1": input1,
                "input2": input2,
                "dtype1": dtype1,
                "dtype2": dtype2,
            },
            rtol=tolerance,
            atol=tolerance,
        )

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.parametrize("dtype1,dtype2", [("int32", "float32"), ("int64", "float64")])
    def test_logaddexp2_integer_promotion(self, dtype1, dtype2, ie_device, precision, ir_version):
        input1 = np.array([[0, 2], [-2, 5]], dtype=dtype1)
        input2 = np.array([0.5, 2.25], dtype=dtype2)
        self._test(
            *self.create_model(),
            ie_device,
            precision,
            ir_version,
            kwargs_to_prepare_input={
                "input1": input1,
                "input2": input2,
                "dtype1": dtype1,
                "dtype2": dtype2,
            },
            rtol=1e-5,
            atol=1e-6,
        )
