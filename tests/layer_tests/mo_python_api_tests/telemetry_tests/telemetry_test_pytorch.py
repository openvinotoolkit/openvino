# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import sys
import tempfile
import unittest
from openvino.tools.mo import convert_model
from openvino.tools.mo.utils.version import get_simplified_mo_version, get_simplified_ie_version
from test_mo_convert_pytorch import make_pt_model_two_inputs
from unittest.mock import MagicMock, call, patch

try:
    import openvino_telemetry as tm
except ImportError:
    import openvino.tools.mo.utils.telemetry_stub as tm


class TestGeneralTelemetrySending(unittest.TestCase):
    def test_general_telemetry_sending(self):
        tm.Telemetry.send_event = MagicMock()

        message = str(dict({
            "platform": platform.system(),
            "mo_version": get_simplified_mo_version(),
            "ie_version": get_simplified_ie_version(env=os.environ),
            "python_version": sys.version,
            "return_code": 0
        }))

        # Test PyTorch
        model = make_pt_model_two_inputs()

        _ = convert_model(model)

        calls = [call('mo', 'version', get_simplified_mo_version()),
                 call('mo', 'cli_parameters', 'framework:pytorch'),
                 call('mo', 'cli_parameters', 'input_model:1'),
                 call('mo', 'op_count', 'pytorch_aten::add', 1),
                 call('mo', 'op_count', 'pytorch_aten::relu', 1),
                 call('mo', 'op_count', 'pytorch_aten::sigmoid', 1),
                 call('mo', 'op_count', 'pytorch_prim::Constant', 1),
                 call('mo', 'conversion_method', 'pytorch_frontend'),
                 call('mo', 'framework', 'pytorch'),
                 call('mo', 'offline_transformations_status', message),
                 call('mo', 'conversion_result', 'success')]

        tm.Telemetry.send_event.assert_has_calls(calls)
        tm.Telemetry.send_event.reset_mock()

