# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from openvino.runtime import Layout, Dimension

from openvino.tools.mo import LayoutMap
from openvino.tools.mo.utils.cli_parser import mean_scale_value_to_str, \
    transform_param_to_str, str_list_to_str, source_target_layout_to_str, layout_param_to_str
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class TestConvertingConvertArgumentsToString(UnitTestWithMockedTelemetry):
    def test_mean_scale_value_to_str(self):
        values = [0.5, 1.3, 0.67]
        self.assertTrue(mean_scale_value_to_str(values) == "[0.5,1.3,0.67]")

        values = {"input": [0.5, 1.3, 0.67]}
        self.assertTrue(mean_scale_value_to_str(values) == "input[0.5,1.3,0.67]")

        values = {"input1": [0.5, 1.3, 0.67], "input2": [4.2, 6.7, 3.15], "input3": [0.757, 4.6, 7.3]}
        self.assertTrue(mean_scale_value_to_str(values) ==
                        "input1[0.5,1.3,0.67],input2[4.2,6.7,3.15],input3[0.757,4.6,7.3]")

        self.assertRaises(Exception, mean_scale_value_to_str, **{"value": {("a", "b"): [0.5, 1.3, 0.67]}})
        self.assertRaises(Exception, mean_scale_value_to_str, **{"value": {"name": Dimension(1)}})
        self.assertRaises(Exception, mean_scale_value_to_str, **{"value": Dimension(1)})

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

    def test_source_target_layout_to_str(self):
        layout = {"input1": Layout("nhwc"), "input2": Layout("n??"), "input3": "nchw"}
        self.assertTrue(source_target_layout_to_str(layout) == "input1([N,H,W,C]),input2([N,?,?]),input3(nchw)")

        self.assertRaises(Exception, source_target_layout_to_str, **{"value": {"op": Dimension(1)}})
        self.assertRaises(Exception, source_target_layout_to_str, **{"value": {("a", "b"): Layout("nhwc")}})
        self.assertRaises(Exception, source_target_layout_to_str, **{"value": Dimension(1)})

    def test_layout_param_to_str_to_str(self):
        layout = {"input1": Layout("nhwc"), "input2": Layout("n??"), "input3": "nchw"}
        self.assertTrue(layout_param_to_str(layout) == "input1([N,H,W,C]),input2([N,?,?]),input3(nchw)")

        layout_map1 = LayoutMap(source_layout=Layout("n??"), target_layout=None)
        layout_map2 = LayoutMap(source_layout=Layout("nhwc"), target_layout=("nchw"))
        layout_map3 = LayoutMap(source_layout="abc", target_layout="cab")

        layout = {"input1": layout_map1, "input2": layout_map2, "input3": layout_map3, "input4": Layout("nhwc"),
                  "input5": "n?"}

        self.assertTrue(layout_param_to_str(layout) == "input1([N,?,?]),input2([N,H,W,C]->nchw),"
                                                       "input3(abc->cab),input4([N,H,W,C]),input5(n?)")

        self.assertRaises(Exception, layout_param_to_str, **{"value": {"op": Dimension(1)}})
        self.assertRaises(Exception, layout_param_to_str, **{"value": {("a", "b"): Layout("nhwc")}})
        self.assertRaises(Exception, layout_param_to_str, **{"value": Dimension(1)})

        layout = ["nhwc", "[n,c]"]
        self.assertTrue(layout_param_to_str(layout) == "nhwc,[n,c]")

        layout = ["abc->cab", "..nc"]
        self.assertTrue(layout_param_to_str(layout) == "abc->cab,..nc")

        layout_map1 = LayoutMap(source_layout=Layout("n??"), target_layout=None)
        layout = [layout_map1, "..nc"]
        self.assertTrue(layout_param_to_str(layout) == "[N,?,?],..nc")

        layout_map2 = LayoutMap(source_layout=Layout("nhwc"), target_layout=("nchw"))
        layout_map3 = LayoutMap(source_layout="abc", target_layout="cab")
        layout = [layout_map2, layout_map3]
        self.assertTrue(layout_param_to_str(layout) == "[N,H,W,C]->nchw,abc->cab")
