# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import unittest
from unittest.mock import patch


class TestTelemetryFallback(unittest.TestCase):
    def test_telemetry_fallback_stub(self):
        """Verify that when openvino_telemetry is absent, ovc modules fall back to telemetry_stub."""
        with patch.dict(sys.modules, {"openvino_telemetry": None, "openvino_telemetry.backend": None}):
            from openvino.tools.ovc import telemetry_utils, convert_impl, utils
            importlib.reload(telemetry_utils)
            importlib.reload(convert_impl)
            importlib.reload(utils)

            import openvino.tools.ovc.telemetry_stub as telemetry_stub
            self.assertIs(telemetry_utils.tm, telemetry_stub)
            self.assertIs(convert_impl.tm, telemetry_stub)
            self.assertIs(utils.tm, telemetry_stub)

            t = telemetry_utils.init_ovc_telemetry()
            self.assertIsInstance(t, telemetry_stub.Telemetry)

            # Check that all stub methods can be called safely without exceptions
            t.send_event("ovc", "test", "test")
            t.send_error("ovc", "error_info")
            t.send_stack_trace("ovc", "trace")
            t.start_session("ovc")
            t.end_session("ovc")
            t.force_shutdown(1.0)

    def test_convert_model_without_telemetry(self):
        """Verify that convert_model works without openvino_telemetry installed."""
        with patch.dict(sys.modules, {"openvino_telemetry": None, "openvino_telemetry.backend": None}):
            from openvino.tools.ovc import convert_model, telemetry_utils
            importlib.reload(telemetry_utils)

            import openvino as ov
            param = ov.opset8.parameter([1, 3], dtype=ov.Type.f32)
            relu = ov.opset8.relu(param)
            model = ov.Model([relu], [param])

            converted = convert_model(model)
            self.assertIsNotNone(converted)

    def test_ovc_main_help_without_telemetry(self):
        """Verify that ovc CLI main entrypoint works without openvino_telemetry."""
        with patch.dict(sys.modules, {"openvino_telemetry": None, "openvino_telemetry.backend": None}):
            from openvino.tools.ovc import main
            importlib.reload(main)

            with patch.object(sys, "argv", ["ovc", "--help"]):
                with self.assertRaises(SystemExit) as cm:
                    main.main()
                self.assertEqual(cm.exception.code, 0)
