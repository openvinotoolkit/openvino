# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import tempfile
import unittest

import openvino as ov
from openvino import PartialShape

from openvino.tools.ovc.cli_parser import _InputCutInfo
from openvino.tools.ovc.cli_parser import input_to_input_cut_info, \
     get_all_cli_parser, get_mo_convert_params, parse_inputs, get_model_name_from_args
from openvino.tools.ovc.convert_impl import pack_params_to_args_namespace, arguments_post_parsing, args_to_argv
from openvino.tools.ovc.error import Error
from unit_tests.ovc.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry


class TestShapesParsing(UnitTestWithMockedTelemetry):
    def test_get_shapes_several_inputs_several_shapes2(self):
        # shapes specified using --input command line parameter and no values
        argv_input = "inp1[1,22,333,123],inp2[-1,45,7,1]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([1,22,333,123])),
                      _InputCutInfo(name='inp2', shape=PartialShape([-1,45,7,1]))]
        self.assertEqual(inputs, inputs_ref)

    def test_raises_get_shapes_1(self):
        argv_input = "[h,y]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_raises_get_shapes_2(self):
        argv_input = "(2, 3)"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_raises_get_shapes_3(self):
        argv_input = "input_1(2, 3)"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_raises_get_shapes_4(self):
        argv_input = "(2, 3),(10, 10)"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_raises_get_shapes_5(self):
        argv_input = "<2,3,4>"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_raises_get_shapes_6(self):
        argv_input = "sd<2,3>"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_get_shapes_complex_input(self):
        argv_input = "[10, -1, 100],mask[],[?,?]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(shape=PartialShape([10, -1, 100])),
                      _InputCutInfo(name='mask', shape=PartialShape([])),
                      _InputCutInfo(shape=PartialShape([-1, -1]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_freezing_with_scalar_and_without_shapes_in_input(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1,inp2"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1'),
                      _InputCutInfo(name='inp2')]
        self.assertEqual(inputs, inputs_ref)


    def test_get_shapes_and_freezing_with_scalar(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1,inp2[]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1'),
                      _InputCutInfo(name='inp2', shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_shapes3(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1],inp2[3,2,3],inp3[5]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      _InputCutInfo(name='inp3', shape=PartialShape([5]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_shapes3_comma_sep(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1],inp2[3 2 3],inp3[5]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      _InputCutInfo(name='inp3', shape=PartialShape([5]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_shapes6(self):
        # 0D value for freezing specified using --input command line parameter without shape
        argv_input = "inp1[3,1],inp2[3,2,3],inp3"
        argv_input = parse_inputs(argv_input)
        inputs_list, result, _ = input_to_input_cut_info(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      _InputCutInfo(name='inp3')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_shapes7(self):
        # 0D shape and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1],inp2[3,2,3],inp3[]"
        argv_input = parse_inputs(argv_input)
        inputs_list, result, _ = input_to_input_cut_info(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape([3,2,3])),
                      _InputCutInfo(name='inp3', shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)


    def test_get_shapes_and_data_types_shape_only(self):
        argv_input = "placeholder1[3 1],placeholder2,placeholder3"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='placeholder1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='placeholder2'),
                      _InputCutInfo(name='placeholder3')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_and_data_types_shape_with_ports_only(self):
        argv_input = "placeholder1:4[3 1],placeholder2,2:placeholder3"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='placeholder1:4', shape=PartialShape([3,1])),
                      _InputCutInfo(name='placeholder2'),
                      _InputCutInfo(name='2:placeholder3')]
        self.assertEqual(inputs, inputs_ref)

    def test_wrong_data_types(self):
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{abracadabra},inp3[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_shape_and_value_shape_mismatch(self):
        # size of value tensor does not correspond to specified shape for the third node
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3[5 3]->[2.0 3.0 5.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_get_shapes_no_input_no_shape(self):
        argv_input = ""
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = []
        self.assertEqual(inputs, inputs_ref)


    def test_get_shapes_no_input_one_shape2(self):
        argv_input = "[12,4,1]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(shape=PartialShape([12,4,1]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_for_scalar_inputs(self):
        argv_input = "[]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_two_input_shapes_with_scalar(self):
        argv_input = "[12,4,1],[]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(shape=PartialShape([12,4,1])),
                      _InputCutInfo(shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_two_input_shapes(self):
        argv_input = "[12,4,1],[10]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(shape=PartialShape([12,4,1])),
                      _InputCutInfo(shape=PartialShape([10])),]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_one_input_no_shape(self):
        argv_input = "inp1"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes2(self):
        # shapes specified using --input command line parameter and no values
        argv_input = "inp1[1,?,50..100,123],inp2[-1,45..,..7,1]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape("[1,?,50..100,123]")),
                      _InputCutInfo(name='inp2', shape=PartialShape("[-1,45..,..7,1]"))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes3(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1],inp2[3..,..2,5..10,?,-1],inp3[5]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      _InputCutInfo(name='inp3', shape=PartialShape([5]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes6(self):
        # 0D value for freezing specified using --input command line parameter without shape
        argv_input = "inp1[3 1],inp2[3.. ..2 5..10 ? -1],inp3"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      _InputCutInfo(name='inp3')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes7(self):
        # 0D shape and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1],inp2[3.. ..2 5..10 ? -1],inp3[]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      _InputCutInfo(name='inp3', shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)

    def test_partial_shapes_freeze_dynamic_negative_case1(self):
        argv_input = "inp1:1[3 1..10]->[1.0 2.0 3.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case2(self):
        argv_input = "inp1:1[1 2 -1]->[1.0 2.0 3.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_get_shapes_several_inputs_several_partial_shapes2_comma_separator(self):
        # shapes specified using --input command line parameter and no values
        argv_input = "inp1[1,?,50..100,123],inp2[-1,45..,..7,1]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape("[1,?,50..100,123]")),
                      _InputCutInfo(name='inp2', shape=PartialShape("[-1,45..,..7,1]"))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes3_comma_separator(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1],inp2[3..,..2,5..10,?,-1],inp3[5]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      _InputCutInfo(name='inp3', shape=PartialShape([5]))]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes6_comma_separator(self):
        # 0D value for freezing specified using --input command line parameter without shape
        argv_input = "inp1[3, 1],inp2[3.., ..2, 5..10, ?,-1],inp3"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      _InputCutInfo(name='inp3')]
        self.assertEqual(inputs, inputs_ref)

    def test_get_shapes_several_inputs_several_partial_shapes7_comma_separator(self):
        # 0D shape and value for freezing specified using --input command line parameter
        argv_input = "inp1[3,1],inp2[3.., ..2,5..10, ?,-1],inp3[]"
        argv_input = parse_inputs(argv_input)
        inputs = input_to_input_cut_info(argv_input)
        inputs_ref = [_InputCutInfo(name='inp1', shape=PartialShape([3,1])),
                      _InputCutInfo(name='inp2', shape=PartialShape("[3..,..2,5..10,?,-1]")),
                      _InputCutInfo(name='inp3', shape=PartialShape([]))]
        self.assertEqual(inputs, inputs_ref)

    def test_partial_shapes_freeze_dynamic_negative_case1_comma_separator(self):
        argv_input = "inp1:1[3,1..10]->[1.0 2.0 3.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case2_comma_separator(self):
        argv_input = "inp1:1[1,2,-1]->[1.0 2.0 3.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case3_comma_separator(self):
        argv_input = "inp1:1[3,1..10]->[1.0 2.0 3.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_partial_shapes_freeze_dynamic_negative_case4_comma_separator(self):
        argv_input = "inp1:1[1, 2, -1]->[1.0 2.0 3.0]"
        self.assertRaises(Error, parse_inputs, argv_input)

    def test_not_supported_arrow(self):
        with self.assertRaisesRegex(Exception,
                                    "Incorrect format of input."):
            argv_input = parse_inputs("inp1->[1.0]")
            input_to_input_cut_info(argv_input)


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


class TestPackParamsToArgsNamespace(unittest.TestCase):
    def test_mo_convert_params(self):
        from openvino.frontend import ConversionExtension
        args = {'input_model': os.path.dirname(__file__),
                'extension': ConversionExtension("Ext", lambda x: x),
                'input': ['name', ("a", [1, 2, 3], ov.Type.f32)],
                'output': ["a", "b", "c"]}

        cli_parser = get_all_cli_parser()
        argv = pack_params_to_args_namespace(args, cli_parser, True)

        assert argv.input_model == args['input_model']
        assert argv.extension == args['extension']
        assert argv.input == ['name', ("a", [1,2,3], ov.Type.f32)]
        assert argv.output == ["a", "b", "c"]

        for arg, value in vars(argv).items():
            if arg not in args and arg != 'is_python_api_used':
                assert value == cli_parser.get_default(arg)

    def test_output_post_parsing_1(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': "a,b,c"}

        argv = args_to_argv(**args)

        argv.is_python_api_used = False
        argv = arguments_post_parsing(argv)
        assert argv.output == ["a", "b", "c"]

    def test_output_post_parsing_2(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': "a, b, c"}

        argv = args_to_argv(**args)

        argv.is_python_api_used = False
        argv = arguments_post_parsing(argv)
        assert argv.output == ["a", "b", "c"]

    def test_output_post_parsing_3(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': "a,b, c"}

        argv = args_to_argv(**args)

        argv.is_python_api_used = False
        argv = arguments_post_parsing(argv)
        assert argv.output == ["a", "b", "c"]

    def test_output_post_parsing_4(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': "a , b ,  c"}

        argv = args_to_argv(**args)

        argv.is_python_api_used = False
        argv = arguments_post_parsing(argv)
        assert argv.output == ["a", "b", "c"]

    def test_output_post_parsing_5(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': "a,b"}

        argv = args_to_argv(**args)

        argv.is_python_api_used = True
        argv = arguments_post_parsing(argv)
        assert argv.output == ["a,b"]  # post parsing should decorate single string into a list

    def test_output_post_parsing_6(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': ["first na me", "second name"]}

        argv = args_to_argv(**args)

        argv.is_python_api_used = True
        argv = arguments_post_parsing(argv)
        # when used in python api should be able to handle names with spaces
        assert argv.output == ["first na me", "second name"]

    def test_raises_output_post_parsing_1(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': ["a,b, c", 23]}

        argv = args_to_argv(**args)

        argv.is_python_api_used = True
        self.assertRaises(AssertionError, arguments_post_parsing, argv)

    def test_raises_output_post_parsing_2(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': "na me, full_name"}

        argv = args_to_argv(**args)

        argv.is_python_api_used = False
        with self.assertRaisesRegex(AssertionError, ".*output names should not be empty or contain spaces"):
            arguments_post_parsing(argv)

    def test_raises_output_post_parsing_3(self):
        args = {'input_model': os.path.dirname(__file__),
                'input': "input_1[1,2,3]",
                'output_model': os.getcwd() + "model.xml",
                'output': "a,,b,c"}

        argv = args_to_argv(**args)

        argv.is_python_api_used = False
        with self.assertRaisesRegex(AssertionError, ".*output names should not be empty or contain spaces"):
            arguments_post_parsing(argv)

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
        }

        params = get_mo_convert_params()
        for group_name in ref_params:
            assert group_name in params
            assert params[group_name].keys() == ref_params[group_name]

        cli_parser = get_all_cli_parser()
        for group_name, params in ref_params.items():
            for param_name in params:
                param_name = '--' + param_name
                if param_name in ['--input_model', '--share_weights', '--example_input']:
                    assert param_name not in cli_parser._option_string_actions
                else:
                    assert param_name in cli_parser._option_string_actions



class GetModelNameTest(unittest.TestCase):
    def test_case1(self):
        current_dir = os.getcwd()
        dir = os.path.basename(current_dir)
        argv = argparse.Namespace(input_model="."+ os.sep)
        assert get_model_name_from_args(argv) == current_dir + os.sep + dir + ".xml"

    def test_case2(self):
        current_dir = os.getcwd()
        argv = argparse.Namespace(input_model="."+ os.sep +"test_model")
        assert get_model_name_from_args(argv) == current_dir + os.sep + "test_model.xml"


    def test_case3(self):
        current_dir = os.getcwd()
        argv = argparse.Namespace(input_model="."+ os.sep +"test_model.pb")
        assert get_model_name_from_args(argv) == current_dir + os.sep + "test_model.xml"


    def test_case4(self):
        current_dir = os.getcwd()
        dir = os.path.basename(current_dir)
        argv = argparse.Namespace(input_model="."+ os.sep,
                                  output_model="."+ os.sep)
        assert get_model_name_from_args(argv) == "." + os.sep + dir + ".xml"

    def test_case5(self):
        argv = argparse.Namespace(input_model="test_model.pb",
                                  output_model="."+ os.sep)
        assert get_model_name_from_args(argv) == "." + os.sep + "test_model.xml"


    def test_case6(self):
        argv = argparse.Namespace(input_model="test_model",
                                  output_model="."+ os.sep)
        assert get_model_name_from_args(argv) == "." + os.sep + "test_model.xml"



    def test_case7(self):
        argv = argparse.Namespace(input_model="test_dir" + os.sep,
                                  output_model="."+ os.sep)
        assert get_model_name_from_args(argv) == "." + os.sep + "test_dir.xml"

    def test_case8(self):
        argv = argparse.Namespace(input_model="test_model.pb",
                                  output_model="new_model")
        assert get_model_name_from_args(argv) == "new_model.xml"

    def test_case9(self):
        argv = argparse.Namespace(input_model="test_model",
                                  output_model="new_dir" + os.sep)
        assert get_model_name_from_args(argv) == "new_dir" + os.sep + "test_model.xml"

    def test_case10(self):
        argv = argparse.Namespace(input_model="test_dir" + os.sep,
                                  output_model="new_model.xml")
        assert get_model_name_from_args(argv) == "new_model.xml"

    def test_case11(self):
        argv = argparse.Namespace(input_model="/",
                                  output_model="new_model")
        assert get_model_name_from_args(argv) == "new_model.xml"


    def test_negative(self):

        argv = argparse.Namespace(input_model="/",)
        with self.assertRaisesRegex(Exception, ".*Could not derive model name from input model. Please provide 'output_model' parameter.*"):
            get_model_name_from_args(argv)