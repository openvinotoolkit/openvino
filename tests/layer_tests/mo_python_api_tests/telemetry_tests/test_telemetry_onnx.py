# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import sys
import tempfile
import unittest
from openvino.tools.mo import convert_model
from openvino.tools.mo.utils.version import get_simplified_mo_version, get_simplified_ie_version
from test_mo_convert_extensions import create_onnx_model
from test_mo_convert_pytorch import make_pt_model_two_inputs, create_pt_model_with_custom_op
from unittest.mock import MagicMock, call, patch

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm


class TestGeneralTelemetrySending(unittest.TestCase):
    def test_general_telemetry_sending(self):
        test_directory = os.path.dirname(os.path.realpath(__file__))
        tm.Telemetry.send_event = MagicMock()

        message = str(dict({
            "platform": platform.system(),
            "mo_version": get_simplified_mo_version(),
            "ie_version": get_simplified_ie_version(env=os.environ),
            "python_version": sys.version,
            "return_code": 0
        }))

        # Test ONNX FE
        with tempfile.TemporaryDirectory(dir=test_directory) as tmpdir:
            model = create_onnx_model(tmpdir)
            _ = convert_model(model)

        calls = [call('mo', 'version', get_simplified_mo_version()),
                 call('mo', 'cli_parameters', 'input_model:1'),
                 call('mo', 'op_count', 'onnx_Elu', 1),
                 call('mo', 'op_count', 'onnx_LeakyRelu', 1),
                 call('mo', 'conversion_method', 'onnx_frontend'),
                 call('mo', 'framework', 'onnx'),
                 call('mo', 'offline_transformations_status', message),
                 call('mo', 'conversion_result', 'success')]

        tm.Telemetry.send_event.assert_has_calls(calls)
        tm.Telemetry.send_event.reset_mock()

        # Test ONNX legacy
        with tempfile.TemporaryDirectory(dir=test_directory) as tmpdir:
            model = create_onnx_model(tmpdir)
            _ = convert_model(model, use_legacy_frontend=True)

        calls = [call('mo', 'version', get_simplified_mo_version()),
                 call('mo', 'cli_parameters', 'use_legacy_frontend:True'),
                 call('mo', 'cli_parameters', 'input_model:1'),
                 call('mo', 'framework', 'onnx'),
                 call('mo', 'conversion_method', 'mo_legacy'),
                 call('mo', 'op_count', 'onnx_LeakyRelu', 1),
                 call('mo', 'op_count', 'onnx_Elu', 1),
                 call('mo', 'input_shapes', '{fw:onnx,shape:"[2 3 4]"}'),
                 call('mo', 'partially_defined_shape', '{partially_defined_shape:0,fw:onnx}'),
                 call('mo', 'offline_transformations_status', message),
                 call('mo', 'conversion_result', 'success')]

        tm.Telemetry.send_event.assert_has_calls(calls)
        tm.Telemetry.send_event.reset_mock()
