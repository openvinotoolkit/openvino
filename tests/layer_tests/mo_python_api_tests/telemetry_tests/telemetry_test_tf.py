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
from test_mo_convert_tf import create_keras_model, create_tf_graph_def
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

        # Test TF FE, tf.keras model
        model, _, _ = create_keras_model("")
        _ = convert_model(model)

        calls = [call('mo', 'version', get_simplified_mo_version()),
                 call('mo', 'cli_parameters', 'input_model:1'),
                 call('mo', 'op_count', 'tf_AddV2', 1),
                 call('mo', 'op_count', 'tf_Identity', 1),
                 call('mo', 'op_count', 'tf_Placeholder', 2),
                 call('mo', 'op_count', 'tf_Relu', 1),
                 call('mo', 'op_count', 'tf_Sigmoid', 1),
                 call('mo', 'conversion_method', 'tf_frontend'),
                 call('mo', 'framework', 'tf2'),
                 call('mo', 'offline_transformations_status', message),
                 call('mo', 'conversion_result', 'success')]

        tm.Telemetry.send_event.assert_has_calls(calls)
        tm.Telemetry.send_event.reset_mock()

        # Test TF legacy, tf.keras model
        _ = convert_model(model, use_legacy_frontend=True)

        calls = [call('mo', 'version', get_simplified_mo_version()),
                 call('mo', 'cli_parameters', 'use_legacy_frontend:True'),
                 call('mo', 'cli_parameters', 'input_model:1'),
                 call('mo', 'conversion_method', 'mo_legacy'),
                 call('mo', 'framework', 'tf2'),
                 call('mo', 'op_count', 'tf_Placeholder', 2),
                 call('mo', 'op_count', 'tf_AddV2', 1),
                 call('mo', 'op_count', 'tf_Relu', 1),
                 call('mo', 'op_count', 'tf_Sigmoid', 1),
                 call('mo', 'op_count', 'tf_Identity', 1),
                 call('mo', 'input_shapes', '{fw:tf,shape:"[-1  1  2  3],[-1  1  2  3]"}'),
                 call('mo', 'partially_defined_shape', '{partially_defined_shape:1,fw:tf}'),
                 call('mo', 'offline_transformations_status', message),
                 call('mo', 'conversion_result', 'success')]

        tm.Telemetry.send_event.assert_has_calls(calls)
        tm.Telemetry.send_event.reset_mock()

        # Test TF FE, tf.GraphDef
        model, _, _ = create_tf_graph_def("")
        _ = convert_model(model)

        calls = [call('mo', 'version', 'custom'),
                 call('mo', 'cli_parameters', 'input_model:1'),
                 call('mo', 'op_count', 'tf_AddV2', 1),
                 call('mo', 'op_count', 'tf_NoOp', 1),
                 call('mo', 'op_count', 'tf_Placeholder', 2),
                 call('mo', 'op_count', 'tf_Relu', 1),
                 call('mo', 'op_count', 'tf_Sigmoid', 1),
                 call('mo', 'conversion_method', 'tf_frontend'),
                 call('mo', 'framework', 'tf'),
                 call('mo', 'offline_transformations_status', message),
                 call('mo', 'conversion_result', 'success')]

        tm.Telemetry.send_event.assert_has_calls(calls)
        tm.Telemetry.send_event.reset_mock()

        # Test TF legacy, tf.GraphDef
        _ = convert_model(model, use_legacy_frontend=True)

        calls = [call('mo', 'version', get_simplified_mo_version()),
                 call('mo', 'cli_parameters', 'use_legacy_frontend:True'),
                 call('mo', 'cli_parameters', 'input_model:1'),
                 call('mo', 'conversion_method', 'mo_legacy'),
                 call('mo', 'framework', 'tf'),
                 call('mo', 'op_count', 'tf_Placeholder', 2),
                 call('mo', 'op_count', 'tf_AddV2', 1),
                 call('mo', 'op_count', 'tf_Relu', 1),
                 call('mo', 'op_count', 'tf_Sigmoid', 1),
                 call('mo', 'op_count', 'tf_NoOp', 1),
                 call('mo', 'input_shapes', '{fw:tf,shape:"[1 2 3],[1 2 3]"}'),
                 call('mo', 'partially_defined_shape', '{partially_defined_shape:0,fw:tf}'),
                 call('mo', 'offline_transformations_status', message),
                 call('mo', 'conversion_result', 'success')]

        tm.Telemetry.send_event.assert_has_calls(calls)
        tm.Telemetry.send_event.reset_mock()
