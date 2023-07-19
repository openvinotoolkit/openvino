# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openvino.runtime import Layout, Dimension

from openvino.tools.ovc.cli_parser import str_list_to_str
from unit_tests.ovc.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class TestConvertingConvertArgumentsToString(UnitTestWithMockedTelemetry):
    def test_str_list_to_str(self):
        list_str = ["data1", "data2", "data3"]
        self.assertTrue(str_list_to_str(list_str) == "data1,data2,data3")

        list_str = "data1"
        self.assertTrue(str_list_to_str(list_str) == "data1")

        self.assertRaises(Exception, str_list_to_str, **{"values": [int, 1]})
        self.assertRaises(Exception, str_list_to_str, **{"values": Dimension(1)})
