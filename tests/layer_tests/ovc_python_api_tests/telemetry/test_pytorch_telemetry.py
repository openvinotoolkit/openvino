# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import MagicMock, patch

import openvino_telemetry as tm
import torch
from openvino.tools.ovc import convert_model


def arange_pt_model():

    class aten_arange_end_dtype(torch.nn.Module):
        def __init__(self, dtype) -> None:
            super(aten_arange_end_dtype, self).__init__()
            self.dtype = dtype

        def forward(self, x):
            return torch.arange(x, dtype=self.dtype)

    return aten_arange_end_dtype(torch.float32)


def mocked_inputs(self):
    # This line returns incorrect inputs and causes exception raise in translator
    if hasattr(self, "graph_element") and hasattr(self.graph_element, "kind") and self.graph_element.kind() == "aten::arange":
        return [0]

    return [x.unique() for x in self.raw_inputs]


@patch('openvino.frontend.pytorch.ts_decoder.TorchScriptPythonDecoder.inputs', mocked_inputs)
class TestGeneralTelemetrySending(unittest.TestCase):
    def test_general_telemetry_sending(self):
        tm.Telemetry.send_event = MagicMock()

        # Create PyTorch model with Arange
        model = torch.jit.script(arange_pt_model())

        try:
            _ = convert_model(model, input=[torch.float32])
        except:
            pass

        tm.Telemetry.send_event.assert_any_call('ovc', 'error_info', '[PyTorch Frontend] Not expected number of inputs for aten::arange\n', 1)
        tm.Telemetry.send_event.reset_mock()
