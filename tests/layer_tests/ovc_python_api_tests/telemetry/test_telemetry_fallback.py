# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import unittest
from unittest.mock import patch


class TestTelemetryFallback(unittest.TestCase):
    def tearDown(self):
        try:
            from openvino.tools.ovc import telemetry_utils
            importlib.reload(telemetry_utils)
        except Exception:
            pass

    def test_telemetry_stub_methods(self):
        """Verify that all telemetry_stub methods execute safely without raising exceptions."""
        from openvino.tools.ovc.telemetry_stub import Telemetry

        t = Telemetry("ovc", "1.0")
        t.send_event("ovc", "test_action", "test_label", 1)
        t.send_error("ovc", "error_info")
        t.send_stack_trace("ovc", "test_trace")
        t.start_session("ovc")
        t.end_session("ovc")
        t.force_shutdown(1.0)

    def test_telemetry_fallback_stub(self):
        """Verify that when openvino_telemetry is absent, telemetry_utils uses telemetry_stub."""
        with patch.dict(sys.modules, {"openvino_telemetry": None, "openvino_telemetry.backend": None}):
            from openvino.tools.ovc import telemetry_utils
            importlib.reload(telemetry_utils)

            import openvino.tools.ovc.telemetry_stub as telemetry_stub
            self.assertIs(telemetry_utils.tm, telemetry_stub)

            t = telemetry_utils.tm.Telemetry()
            self.assertIsInstance(t, telemetry_stub.Telemetry)
            telemetry_utils.send_framework_info("framework")
            telemetry_utils.send_conversion_result("success", need_shutdown=True)
