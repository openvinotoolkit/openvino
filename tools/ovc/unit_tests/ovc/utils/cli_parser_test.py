# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import numpy
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from openvino.tools.ovc.cli_parser import input_to_input_cut_info, \
    check_positive, writable_dir, \
    readable_file_or_object, get_freeze_placeholder_values, get_all_cli_parser, \
    get_mo_convert_params
from openvino.tools.ovc.convert_impl import pack_params_to_args_namespace
from openvino.tools.ovc.error import Error
from unit_tests.ovc.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry
from openvino.runtime import PartialShape, Dimension, Layout
from openvino.tools.ovc import InputCutInfo


class TestShapesParsing(UnitTestWithMockedTelemetry):
    def test_get_shapes_several_inputs_several_shapes2(self):
        # shapes specified using --input command line parameter and no values
        argv_input = "inp1[1,22,333,123],inp2[-1,45,7,1]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([1,22,333,123])),
                      InputCutInfo(name='inp2', shape=PartialShape([-1,45,7,1]))]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_and_freezing_with_scalar_and_without_shapes_in_input(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1,inp2->157"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1'),
                      InputCutInfo(name='inp2', value=157)]
        self.assertEqual(inputs, inputs_ref)

        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp2': 157}

        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            self.assertEqual(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_and_freezing_with_scalar(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1,inp2[]->157"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1'),
                      InputCutInfo(name='inp2', shape=PartialShape([]), value=157)]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_shapes3(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3,2,3],inp3[5]->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      InputCutInfo(name='inp3', shape=PartialShape([5]), value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']),
                                  'inp3': np.array(['1.0', '1.0', '2.0', '3.0', '5.0'])}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_shapes3_comma_sep(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3[5]->[1.0, 1.0, 2.0, 3.0,5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      InputCutInfo(name='inp3', shape=PartialShape([5]), value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']),
                                  'inp3': np.array(['1.0', '1.0', '2.0', '3.0', '5.0'])}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_shapes6(self):
        # 0D value for freezing specified using --input command line parameter without shape
        argv_input = "inp1[3,1]->[1.0 2.0 3.0],inp2[3,2,3],inp3->False"
        inputs_list, result, _ = input_to_input_cut_info(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      InputCutInfo(name='inp3', value=False)]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': False}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_shapes7(self):
        # 0D shape and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1]->[1.0 2.0 3.0],inp2[3,2,3],inp3[]->True"
        inputs_list, result, _ = input_to_input_cut_info(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      InputCutInfo(name='inp3', shape=PartialShape([]), value=True)]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': True}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_and_data_types1(self):
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{i32},inp3[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3]), type=np.int32),
                      InputCutInfo(name='inp3', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_with_input_ports(self):
        argv_input = "1:inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{i32},0:inp3[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='1:inp1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3]), type=np.int32),
                      InputCutInfo(name='0:inp3', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_with_output_ports(self):
        argv_input = "inp1:1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{i32},inp3:4[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1:1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3]), type=np.int32),
                      InputCutInfo(name='inp3:4', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_with_output_ports_comma_sep(self):
        argv_input = "inp1:1[3,1]->[1.0,2.0 ,3.0],inp2[3,2, 3]{i32},inp3:4[5]{f32}->[1.0, 1.0,2.0, 3.0,5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1:1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2', shape=PartialShape([3,2,3]), type=np.int32),
                      InputCutInfo(name='inp3:4', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_shape_only(self):
        argv_input = "placeholder1[3 1],placeholder2,placeholder3"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='placeholder1', shape=PartialShape([3,1])),
                      InputCutInfo(name='placeholder2'),
                      InputCutInfo(name='placeholder3')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_shape_with_ports_only(self):
        argv_input = "placeholder1:4[3 1],placeholder2,2:placeholder3"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='placeholder1:4', shape=PartialShape([3,1])),
                      InputCutInfo(name='placeholder2'),
                      InputCutInfo(name='2:placeholder3')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_when_no_freeze_value(self):
        argv_input = "placeholder1{i32}[3 1],placeholder2,placeholder3{i32}"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='placeholder1', shape=PartialShape([3,1]), type=np.int32),
                      InputCutInfo(name='placeholder2'),
                      InputCutInfo(name='placeholder3', type=np.int32)]
        self.assertEqual(inputs, inputs_ref)

    def test_wrong_data_types(self):
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{abracadabra},inp3[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)

    def test_shape_and_value_shape_mismatch(self):
        # size of value tensor does not correspond to specified shape for the third node
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3[5 3]->[2.0 3.0 5.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)

    def test_wrong_data_for_input_cmd_param(self):
        # test that wrongly formatted data specified in --input is handled properly
        argv_input = "abc->[1.0"
        self.assertRaises(Error, get_freeze_placeholder_values, argv_input)
        argv_input = "def[2 2]->[1.0 2.0 3.0 4.0],abc->1.0 34]"
        self.assertRaises(Error, get_freeze_placeholder_values, argv_input)

    def test_get_shapes_no_input_no_shape(self):
        argv_input = ""
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = []
        self.assertEqual(inputs, inputs_ref)


    def test_get_shapes_no_input_one_shape2(self):
        argv_input = "[12,4,1]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(shape=PartialShape([12,4,1]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_for_scalar_inputs(self):
        argv_input = "[]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_two_input_shapes_with_scalar(self):
        argv_input = "[12,4,1],[]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(shape=PartialShape([12,4,1])),
                      InputCutInfo(shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_two_input_shapes(self):
        argv_input = "[12,4,1],[10]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(shape=PartialShape([12,4,1])),
                      InputCutInfo(shape=PartialShape([10])),]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_one_input_no_shape(self):
        argv_input = "inp1"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes2(self):
        # shapes specified using --input command line parameter and no values
        argv_input = "inp1[1,?,50..100,123],inp2[-1,45..,..7,1]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape("[1,?,50..100,123]")),
                      InputCutInfo(name='inp2', shape=PartialShape("[-1,45..,..7,1]"))]
        self.assertEqual(inputs, inputs_ref)

        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_partial_shapes3(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1]->[1.0 2.0 3.0],inp2[3..,..2,5..10,?,-1],inp3[5]->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=["1.0", "2.0", "3.0"]),
                      InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      InputCutInfo(name='inp3', shape=PartialShape([5]), value=["1.0", "1.0", "2.0", "3.0", "5.0"])]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': np.array(['1.0', '1.0', '2.0', '3.0', '5.0'])}
        input_node_names_ref = "inp1,inp2,inp3"
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_partial_shapes6(self):
        # 0D value for freezing specified using --input command line parameter without shape
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3.. ..2 5..10 ? -1],inp3->False"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=["1.0", "2.0", "3.0"]),
                      InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      InputCutInfo(name='inp3', value=False)]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': False}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_partial_shapes7(self):
        # 0D shape and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3.. ..2 5..10 ? -1],inp3[]->True"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=["1.0", "2.0", "3.0"]),
                      InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      InputCutInfo(name='inp3', shape=PartialShape([]), value=True)]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': True}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_and_data_types_partial_shape_with_input_port(self):
        argv_input = "inp1:1[3 1]->[1.0 2.0 3.0],0:inp2[3.. ..2 5..10 ? -1]{i32},inp3:4[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1:1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='0:inp2', shape=PartialShape("[3..,..2,5..10,?,-1]"), type=np.int32),
                      InputCutInfo(name='inp3:4', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_partial_shape_with_output_port(self):
        argv_input = "inp1:1[3 1]->[1.0 2.0 3.0],inp2:3[3.. ..2 5..10 ? -1]{i32},inp3:4[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1:1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2:3', shape=PartialShape("[3..,..2,5..10,?,-1]"), type=np.int32),
                      InputCutInfo(name='inp3:4', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_partial_shapes_freeze_dynamic_negative_case1(self):
        argv_input = "inp1:1[3 1..10]->[1.0 2.0 3.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case2(self):
        argv_input = "inp1:1[1 2 -1]->[1.0 2.0 3.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)

    def test_get_shapes_several_inputs_several_partial_shapes2_comma_separator(self):
        # shapes specified using --input command line parameter and no values
        argv_input = "inp1[1,?,50..100,123],inp2[-1,45..,..7,1]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape("[1,?,50..100,123]")),
                      InputCutInfo(name='inp2', shape=PartialShape("[-1,45..,..7,1]"))]
        self.assertEqual(inputs, inputs_ref)

        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_partial_shapes3_comma_separator(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1]->[1.0 2.0 3.0],inp2[3..,..2,5..10,?,-1],inp3[5]->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=["1.0", "2.0", "3.0"]),
                      InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      InputCutInfo(name='inp3', shape=PartialShape([5]), value=["1.0", "1.0", "2.0", "3.0", "5.0"])]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']),
                                  'inp3': np.array(['1.0', '1.0', '2.0', '3.0', '5.0'])}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_partial_shapes6_comma_separator(self):
        # 0D value for freezing specified using --input command line parameter without shape
        argv_input = "inp1[3, 1]->[1.0 2.0 3.0],inp2[3.., ..2, 5..10, ?,-1],inp3->False"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=["1.0", "2.0", "3.0"]),
                      InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      InputCutInfo(name='inp3', value=False)]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': False}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_partial_shapes7_comma_separator(self):
        # 0D shape and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1]->[1.0 2.0 3.0],inp2[3.., ..2,5..10, ?,-1],inp3[]->True"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1', shape=PartialShape([3,1]), value=["1.0", "2.0", "3.0"]),
                      InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      InputCutInfo(name='inp3', shape=PartialShape([]), value=True)]
        self.assertEqual(inputs, inputs_ref)
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': True}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            assert np.array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_and_data_types_partial_shape_with_input_port_comma_separator(self):
        argv_input = "inp1:1[3,1]->[1.0 2.0 3.0],0:inp2[ 3.. ,..2, 5..10, ?,-1]{i32},inp3:4[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1:1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='0:inp2', shape=PartialShape("[3..,..2,5..10,?,-1]"), type=np.int32),
                      InputCutInfo(name='inp3:4', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_partial_shape_with_output_port_comma_separator(self):
        argv_input = "inp1:1[3,1]->[1.0 2.0 3.0],inp2:3[3..,..2,5..10,?,-1]{i32},inp3:4[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [InputCutInfo(name='inp1:1', shape=PartialShape([3,1]), value=['1.0', '2.0', '3.0']),
                      InputCutInfo(name='inp2:3', shape=PartialShape("[3..,..2,5..10,?,-1]"), type=np.int32),
                      InputCutInfo(name='inp3:4', shape=PartialShape([5]), type=np.float32, value=['1.0', '1.0', '2.0', '3.0', '5.0'])]
        self.assertEqual(inputs, inputs_ref)

    def test_partial_shapes_freeze_dynamic_negative_case1_comma_separator(self):
        argv_input = "inp1:1[3,1..10]->[1.0 2.0 3.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case2_comma_separator(self):
        argv_input = "inp1:1[1,2,-1]->[1.0 2.0 3.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case3_comma_separator(self):
        argv_input = "inp1:1[3,1..10]->[1.0 2.0 3.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case4_comma_separator(self):
        argv_input = "inp1:1[1, 2, -1]->[1.0 2.0 3.0]"
        self.assertRaises(Error, input_to_input_cut_info, argv_input)


class PositiveChecker(unittest.TestCase):
    def test_positive_checker_batch(self):
        res = check_positive('1')
        self.assertEqual(res, 1)

    def test_positive_checker_batch_negative(self):
        self.assertRaises(argparse.ArgumentTypeError, check_positive, '-1')

    def test_positive_checker_batch_not_int(self):
        self.assertRaises(argparse.ArgumentTypeError, check_positive, 'qwe')


class PathCheckerFunctions(unittest.TestCase):
    READABLE_DIR = tempfile.gettempdir()
    WRITABLE_DIR = os.path.join(tempfile.gettempdir(), 'writable_dir')
    WRITABLE_NON_EXISTING_DIR = os.path.join(WRITABLE_DIR, 'non_existing_dir')
    NOT_WRITABLE_DIR = os.path.join(tempfile.gettempdir(), 'not_writable_dir')
    NOT_WRITABLE_SUB_DIR = os.path.join(tempfile.gettempdir(), 'another_not_writable_dir', 'not_existing_dir')
    EXISTING_FILE = tempfile.NamedTemporaryFile(mode='r+', delete=False).name
    NOT_EXISTING_FILE = '/abcd/efgh/ijkl'

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(__class__.WRITABLE_DIR):
            os.makedirs(__class__.WRITABLE_DIR)
        if os.path.exists(__class__.WRITABLE_NON_EXISTING_DIR):
            os.removedirs(__class__.WRITABLE_NON_EXISTING_DIR)

        if not os.path.exists(__class__.NOT_WRITABLE_DIR):
            os.makedirs(__class__.NOT_WRITABLE_DIR)
        os.chmod(__class__.NOT_WRITABLE_DIR, 0)

        if not os.path.exists(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR)):
            os.makedirs(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR))
        os.chmod(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR), 0)
        if os.path.exists(__class__.NOT_EXISTING_FILE):
            os.remove(__class__.NOT_EXISTING_FILE)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(__class__.WRITABLE_DIR):
            os.removedirs(__class__.WRITABLE_DIR)
        if os.path.exists(__class__.NOT_WRITABLE_DIR):
            shutil.rmtree(__class__.NOT_WRITABLE_DIR, ignore_errors=True)
        if os.path.exists(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR)):
            shutil.rmtree(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR), ignore_errors=True)
        if os.path.exists(__class__.EXISTING_FILE):
            os.remove(__class__.EXISTING_FILE)

    def test_single_writable_dir(self):
        self.assertEqual(__class__.WRITABLE_DIR, writable_dir(__class__.WRITABLE_DIR))

    @unittest.skip("Temporary disabled since chmod() is temporary not working on Linux. (Windows do not support not writable dir at all)")
    def test_single_non_writable_dir(self):
        with self.assertRaises(Error) as cm:
            writable_dir(__class__.NOT_WRITABLE_DIR)

    @unittest.skip("Temporary disabled since chmod() is temporary not working on Linux. (Windows do not support not writable dir at all)")
    def test_single_non_writable_sub_dir(self):
        with self.assertRaises(Error) as cm:
            writable_dir(__class__.NOT_WRITABLE_SUB_DIR)

    def test_multiple_writable_dirs(self):
        dirs_str = ','.join([__class__.WRITABLE_DIR, __class__.WRITABLE_NON_EXISTING_DIR])
        self.assertEqual(dirs_str, writable_dir(dirs_str))

    def test_single_writable_non_existing_dir(self):
        self.assertEqual(__class__.WRITABLE_NON_EXISTING_DIR, writable_dir(__class__.WRITABLE_NON_EXISTING_DIR))


    def test_readable_file(self):
        self.assertEqual(__class__.EXISTING_FILE, readable_file_or_object(__class__.EXISTING_FILE))

    def test_non_readable_file(self):
        with self.assertRaises(Error) as cm:
            readable_file_or_object(__class__.NOT_EXISTING_FILE)


class TestPackParamsToArgsNamespace(unittest.TestCase):
    def test_mo_convert_params(self):
        from openvino.frontend import ConversionExtension
        args = {'input_model': os.path.dirname(__file__),
                'extension': ConversionExtension("Ext", lambda x: x),
                'input': ['name', InputCutInfo("a", [1,2,3], numpy.float32, [5, 6, 7])],
                'output': ["a", "b", "c"]}

        cli_parser = get_all_cli_parser()
        argv = pack_params_to_args_namespace(args, cli_parser, True)

        assert argv.input_model == args['input_model']
        assert argv.extension == args['extension']
        assert argv.input == ['name', InputCutInfo("a", [1,2,3], numpy.float32, [5, 6, 7])]
        assert argv.output == ["a","b","c"]

        for arg, value in vars(argv).items():
            if arg not in args and arg != 'is_python_api_used':
                assert value == cli_parser.get_default(arg)

    def test_not_existing_dir(self):
        args = {"input_model": "abc"}
        cli_parser = get_all_cli_parser()

        with self.assertRaisesRegex(Error, "The value for parameter \"input_model\" must be existing file/directory, "
                                           "but \"abc\" does not exist."):
            pack_params_to_args_namespace(args, cli_parser, True)

    def test_unknown_params(self):
        args = {"input_model": os.path.dirname(__file__),
                "a": "b"}
        cli_parser = get_all_cli_parser()

        with self.assertRaisesRegex(Error, "Unrecognized argument: a"):
            pack_params_to_args_namespace(args, cli_parser, True)


class TestConvertModelParamsParsing(unittest.TestCase):
    def test_mo_convert_params_parsing(self):
        ref_params = {
            'Optional parameters:': {'input_model', 'input', 'output', 'example_input',
                                               'extension', 'verbose', 'share_weights'},
            'PaddlePaddle-specific parameters:': {'example_output'},
        }

        params = get_mo_convert_params()
        for group_name in ref_params:
            assert group_name in params
            assert params[group_name].keys() == ref_params[group_name]

        cli_parser = get_all_cli_parser()
        for group_name, params in ref_params.items():
            for param_name in params:
                param_name = '--' + param_name
                if group_name == 'PaddlePaddle-specific parameters:' or \
                        param_name in ['--input_model', '--share_weights', '--example_input']:
                    assert param_name not in cli_parser._option_string_actions
                else:
                    assert param_name in cli_parser._option_string_actions

