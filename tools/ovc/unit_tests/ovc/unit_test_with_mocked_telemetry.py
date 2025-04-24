# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock

try:
    import openvino_telemetry as tm
    from openvino_telemetry.backend import backend_ga4
except ImportError:
    import openvino.tools.ovc.telemetry_stub as tm


class UnitTestWithMockedTelemetry(unittest.TestCase):
    def setUp(self):
        tm.Telemetry.__init__ = Mock(return_value=None)
        tm.Telemetry.send_event = Mock()
        tm.Telemetry.start_session = Mock()
        tm.Telemetry.end_session = Mock()
        tm.Telemetry.force_shutdown = Mock()
