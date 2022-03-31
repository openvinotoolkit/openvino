
import argparse
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openvino.tools.mo.utils.cli_parser import get_placeholder_shapes, get_tuple_values, get_mean_scale_dictionary, \
    get_model_name, \
    parse_tuple_pairs, check_positive, writable_dir, readable_dirs, \
    readable_file, get_freeze_placeholder_values, parse_transform, check_available_transforms, get_layout_values, get_data_type_from_input_value
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from openvino.tools.mo.convert import input_to_str, InputCutInfo, mean_scale_value_to_str



class TestConvertingConvertArgumentsToString(UnitTestWithMockedTelemetry):
    def test_input_to_str(self):
        inp1 = InputCutInfo("data:0")
        self.assertTrue(input_to_str(inp1) == "data:0")

        inp2 = InputCutInfo("data:0", [1, 3, 100, 100])
        self.assertTrue(input_to_str(inp2) == "data:0[1 3 100 100]")

        inp3 = InputCutInfo("data:0", type=np.int32)
        self.assertTrue(input_to_str(inp3) == "data:0{i32}")

        inp4 = InputCutInfo("data:0", value=[2, 4, 5])
        self.assertTrue(input_to_str(inp4) == "data:0->[2 4 5]")

        inp5 = InputCutInfo("data:0", [1, 3, 100, 100], np.uint8)
        self.assertTrue(input_to_str(inp5) == "data:0[1 3 100 100]{u8}")

        inp6 = InputCutInfo("data:0", [2, 5, 7], value=[1, 2, 3, 4, 5])
        self.assertTrue(input_to_str(inp6) == "data:0[2 5 7]->[1 2 3 4 5]")

        inp7 = InputCutInfo("0:data1", type=np.float64, value=[1.6, 7.2, 5.66])
        self.assertTrue(input_to_str(inp7) == "0:data1{f64}->[1.6 7.2 5.66]")

        inp8 = InputCutInfo("data2", [4, 5, 6], np.int64, [5, 4, 3, 2, 1])
        self.assertTrue(input_to_str(inp8) == "data2[4 5 6]{i64}->[5 4 3 2 1]")

        inp = [inp6, inp7, inp8]
        input_to_str(self.assertTrue(input_to_str(inp) == "data:0[2 5 7]->[1 2 3 4 5],"
                                                          "0:data1{f64}->[1.6 7.2 5.66],"
                                                          "data2[4 5 6]{i64}->[5 4 3 2 1]"))

        inp = ["data:0[2 5 7]->[1 2 3 4 5]", "0:data1{f64}->[1.6 7.2 5.66]", "data2[4 5 6]{i64}->[5 4 3 2 1]"]
        input_to_str(self.assertTrue(input_to_str(inp) == "data:0[2 5 7]->[1 2 3 4 5],"
                                                          "0:data1{f64}->[1.6 7.2 5.66],"
                                                          "data2[4 5 6]{i64}->[5 4 3 2 1]"))

    def test_mean_scale_value_to_str(self):
        values = [0.5, 1.3, 0.67]
        input_to_str(self.assertTrue(mean_scale_value_to_str(values) == "[0.5,1.3,0.67]"))

        values = {"input": [0.5, 1.3, 0.67]}
        input_to_str(self.assertTrue(mean_scale_value_to_str(values) == "input[0.5,1.3,0.67]"))

        values = {"input1": [0.5, 1.3, 0.67], "input2": [4.2, 6.7, 3.15], "input3": [0.757, 4.6, 7.3]}
        input_to_str(self.assertTrue(mean_scale_value_to_str(values) ==
                                     "input1[0.5,1.3,0.67],input2[4.2,6.7,3.15],input3[0.757,4.6,7.3]"))

