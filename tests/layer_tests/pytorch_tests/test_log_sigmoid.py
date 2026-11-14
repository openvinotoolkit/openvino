# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from pytorch_layer_test_class import PytorchLayerTest


class TestLogSigmoidAccuracy(PytorchLayerTest):
    def _prepare_input(self):
        values = np.array([-1000, -100, -87.5, -20, -1, -0.0, 0, 1, 5, 10, 16, 20, 50, 80], np.float32)
        return (np.concatenate((values, np.linspace(-100, 80, 2001, dtype=np.float32))),)

    def create_model(self, dtype):
        class LogSigmoidAccuracy(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                x = x.to(self.dtype)
                result = F.logsigmoid(x)
                # The positive tail approaches -exp(-x). Normalize it so the
                # absolute tolerance cannot hide a result rounded to zero.
                scale = torch.exp(-torch.clamp_min(x, 0))
                return result / scale

        return LogSigmoidAccuracy(dtype), "aten::log_sigmoid"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("dynamic_shapes", [False, True])
    @pytest.mark.parametrize("trace_model", [False, True])
    def test_log_sigmoid_accuracy(self, dtype, dynamic_shapes, trace_model, ie_device, precision, ir_version):
        if precision == "FP16":
            pytest.skip("The normalized tail requires the f32 exponent range")
        if self.use_torch_export() and trace_model:
            pytest.skip("trace_model only selects a TorchScript conversion route")
        export_shapes = ({0: torch.export.Dim("length", min=1)},) if dynamic_shapes else {}
        self._test(*self.create_model(dtype), ie_device, precision, ir_version,
                   dynamic_shapes=dynamic_shapes, trace_model=trace_model, custom_eps=2e-5,
                   dynamic_shapes_for_export=export_shapes)


class TestLogSigmoidSpecialValues(PytorchLayerTest):
    def _prepare_input(self):
        return (np.array([-np.inf, np.inf, np.nan, -0.0, 0.0], np.float32),)

    def create_model(self, dtype):
        class LogSigmoidSpecialValues(torch.nn.Module):
            def __init__(self, dtype):
                super().__init__()
                self.dtype = dtype

            def forward(self, x):
                return F.logsigmoid(x.to(self.dtype))

        return LogSigmoidSpecialValues(dtype), "aten::log_sigmoid"

    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.precommit_torch_export
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_log_sigmoid_special_values(self, dtype, ie_device, precision, ir_version):
        if ie_device == "CPU" and dtype == torch.float64:
            pytest.skip("CPU f64-to-f32 precision conversion saturates infinities")
        self._test(*self.create_model(dtype), ie_device, precision, ir_version)
