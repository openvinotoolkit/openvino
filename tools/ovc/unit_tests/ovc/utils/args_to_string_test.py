# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openvino.runtime import Layout, Dimension

from openvino.tools.ovc.cli_parser import transform_param_to_str, str_list_to_str
from unit_tests.ovc.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class TestConvertingConvertArgumentsToString(UnitTestWithMockedTelemetry):
    def test_transform_param_to_str(self):
        transform = 'MakeStateful'
        self.assertTrue(transform_param_to_str(transform) == "MakeStateful")

        transform1 = ('LowLatency2', {'use_const_initializer': False})
        self.assertTrue(transform_param_to_str(transform1) ==
                        "LowLatency2[use_const_initializer=False]")

        transform2 = ('MakeStateful', {'param_res_names': {
            'input_name_1': 'output_name_1', 'input_name_2': 'output_name_2'}})
        self.assertTrue(transform_param_to_str(transform2) ==
                        "MakeStateful[param_res_names={\'input_name_1\':\'output_name_1\',"
                        "\'input_name_2\':\'output_name_2\'}]")

        transform = [transform1, transform2]

        self.assertTrue(transform_param_to_str(transform) == "LowLatency2[use_const_initializer=False],"
                                                             "MakeStateful[param_res_names={"
                                                             "\'input_name_1\':\'output_name_1\',"
                                                             "\'input_name_2\':\'output_name_2\'}]")

        self.assertRaises(Exception, transform_param_to_str, **{"value": ('LowLatency2',
                                                                          {'use_const_initializer': False},
                                                                          "param")})
        self.assertRaises(Exception, transform_param_to_str, **{"value": (("a", "b"), {})})
        self.assertRaises(Exception, transform_param_to_str, **{"value": ('LowLatency2', Dimension(1))})
        self.assertRaises(Exception, transform_param_to_str, **{"value": ('LowLatency2',
                                                                          {('a', 'b'): False})})
        self.assertRaises(Exception, transform_param_to_str, **{"value": Dimension(1)})

    def test_str_list_to_str(self):
        list_str = ["data1", "data2", "data3"]
        self.assertTrue(str_list_to_str(list_str) == "data1,data2,data3")

        list_str = "data1"
        self.assertTrue(str_list_to_str(list_str) == "data1")

        self.assertRaises(Exception, str_list_to_str, **{"values": [int, 1]})
        self.assertRaises(Exception, str_list_to_str, **{"values": Dimension(1)})
