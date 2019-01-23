"""
 Copyright (c) 2018 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import argparse
import imp
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from mo.utils.cli_parser import get_placeholder_shapes, get_tuple_values, get_mean_scale_dictionary, get_model_name, \
    get_absolute_path, parse_tuple_pairs, check_positive, writable_dir, readable_dirs, \
    readable_file
from mo.utils.error import Error


class TestingMeanScaleGetter(unittest.TestCase):
    def test_tuple_parser(self):
        tuple_values = "data(1.1,22.22,333.333),info[2.2,33.33,444.444]"
        result = parse_tuple_pairs(tuple_values)
        exp_res = {
            'data': np.array([1.1, 22.22, 333.333]),
            'info': np.array([2.2, 33.33, 444.444])
        }
        for el in exp_res.keys():
            np.array_equal(result[el], exp_res[el])

    def test_tuple_parser_same_values(self):
        tuple_values = "data(1.1,22.22,333.333),info[1.1,22.22,333.333]"
        result = parse_tuple_pairs(tuple_values)
        exp_res = {
            'data': np.array([1.1, 22.22, 333.333]),
            'info': np.array([1.1, 22.22, 333.333])
        }
        for el in exp_res.keys():
            np.array_equal(result[el], exp_res[el])

    def test_tuple_parser_no_inputs(self):
        tuple_values = "(1.1,22.22,333.333),[2.2,33.33,444.444]"
        result = parse_tuple_pairs(tuple_values)
        exp_res = [np.array([1.1, 22.22, 333.333]),
                   np.array([2.2, 33.33, 444.444])]
        for i in range(0, len(exp_res)):
            np.array_equal(result[i], exp_res[i])

    def test_tuple_parser_error(self):
        tuple_values = "(1.1,22.22,333.333),data[2.2,33.33,444.444]"
        self.assertRaises(Error, parse_tuple_pairs, tuple_values)

    def test_mean_scale_no_input(self):
        mean_values = "data(1.1,22.22,333.333)"
        scale_values = "info[1.1,22.22,333.333]"
        result = get_mean_scale_dictionary(parse_tuple_pairs(mean_values), parse_tuple_pairs(scale_values), None)
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': None
            },
            'info': {
                'mean': None,
                'scale': np.array([1.1, 22.22, 333.333])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_mean_scale_no_input_diff_len(self):
        mean_values = "data(1.1,22.22,333.333),info(2.1,33.22,333.333)"
        scale_values = "info[1.1,22.22,333.333]"
        result = get_mean_scale_dictionary(parse_tuple_pairs(mean_values), parse_tuple_pairs(scale_values), None)
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': None
            },
            'info': {
                'mean': np.array([2.1, 33.22, 333.333]),
                'scale': np.array([1.1, 22.22, 333.333])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_mean_only_input(self):
        mean_values = "data(1.1,22.22,333.333)"
        result = get_mean_scale_dictionary(parse_tuple_pairs(mean_values), parse_tuple_pairs(''), None)
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': None
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_scale_only_input(self):
        scale_values = "data(1.1,22.22,333.333)"
        result = get_mean_scale_dictionary(parse_tuple_pairs(''), parse_tuple_pairs(scale_values), None)
        exp_res = {
            'data': {
                'mean': None,
                'scale': np.array([1.1, 22.22, 333.333])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_scale_only_no_input(self):
        scale_values = "(1.1,22.22,333.333)"
        mean_values = ""
        mean = parse_tuple_pairs(mean_values)
        scale = parse_tuple_pairs(scale_values)
        result = get_mean_scale_dictionary(mean, scale, None)
        exp_res = [
            [
                None,
                np.array([1.1, 22.22, 333.333])
            ]
        ]
        for i in range(len(exp_res)):
            for j in range(len(exp_res[i])):
                if type(exp_res[i][j]) is np.ndarray:
                    np.array_equal(exp_res[i][j], result[i][j])
                else:
                    self.assertEqual(exp_res[i][j], result[i][j])

    def test_scale_only_with_input(self):
        scale_values = "(1.1,22.22,333.333)"
        mean_values = ""
        mean = parse_tuple_pairs(mean_values)
        scale = parse_tuple_pairs(scale_values)
        result = get_mean_scale_dictionary(mean, scale, 'data')
        exp_res = {
            'data': {
                'mean': None,
                'scale': np.array([1.1, 22.22, 333.333])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_2_scale_only_with_input(self):
        scale_values = "(1.1,22.22,333.333),(1.2,22.33,333.444)"
        mean_values = ""
        mean = parse_tuple_pairs(mean_values)
        scale = parse_tuple_pairs(scale_values)
        result = get_mean_scale_dictionary(mean, scale, 'data,info')
        exp_res = {
            'data': {
                'mean': None,
                'scale': np.array([1.1, 22.22, 333.333])
            },
            'info': {
                'mean': None,
                'scale': np.array([1.2, 22.33, 333.444])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_2_mean_only_with_input(self):
        scale_values = ""
        mean_values = "(1.1,22.22,333.333),(1.2,22.33,333.444)"
        mean = parse_tuple_pairs(mean_values)
        scale = parse_tuple_pairs(scale_values)
        result = get_mean_scale_dictionary(mean, scale, 'data,info')
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': None,
            },
            'info': {
                'mean': np.array([1.2, 22.33, 333.444]),
                'scale': None,
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_mean_only_with_input(self):
        scale_values = ""
        mean_values = "(1.1,22.22,333.333)"
        mean = parse_tuple_pairs(mean_values)
        scale = parse_tuple_pairs(scale_values)
        result = get_mean_scale_dictionary(mean, scale, 'data')
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': None
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_mean_scale_diff_no_input(self):
        scale_values = "(1.1,22.22,333.333),(1.1,22.22,333.333)"
        mean_values = "(2.1,11.22,444.333)"
        mean = parse_tuple_pairs(mean_values)
        scale = parse_tuple_pairs(scale_values)
        result = get_mean_scale_dictionary(mean, scale, None)
        exp_res = [
            [
                np.array([2.1, 11.22, 444.333]),  # mean
                np.array([1.1, 22.22, 333.333])  # scale
            ],
            [
                None,  # mean
                np.array([1.1, 22.22, 333.333])  # scale
            ]
        ]
        for i in range(len(exp_res)):
            for j in range(len(exp_res[i])):
                if type(exp_res[i][j]) is np.ndarray:
                    np.array_equal(exp_res[i][j], result[i][j])
                else:
                    self.assertEqual(exp_res[i][j], result[i][j])

    def test_multi_mean_scale_no_input(self):
        mean_values = "data(1.1,22.22,333.333),info(2.1,33.22,444.333)"
        scale_values = "data[1.1,22.22,333.333],info[2.1,33.22,444.333]"
        result = get_mean_scale_dictionary(parse_tuple_pairs(mean_values), parse_tuple_pairs(scale_values), None)
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': np.array([1.1, 22.22, 333.333])
            },
            'info': {
                'mean': np.array([2.1, 33.22, 444.333]),
                'scale': np.array([2.1, 33.22, 444.333])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_multi_mean_scale_input(self):
        mean_values = "data(1.1,22.22,333.333),info(2.1,33.22,444.333)"
        scale_values = "data[1.1,22.22,333.333],info[2.1,33.22,444.333]"
        input_names = 'data,info'
        result = get_mean_scale_dictionary(parse_tuple_pairs(mean_values), parse_tuple_pairs(scale_values), input_names)
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': np.array([1.1, 22.22, 333.333])
            },
            'info': {
                'mean': np.array([2.1, 33.22, 444.333]),
                'scale': np.array([2.1, 33.22, 444.333])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_multi_mean_scale_input_arrays(self):
        mean_values = "(1.1,22.22,333.333),(2.1,33.22,444.333)"
        scale_values = "[1.1,22.22,333.333],[2.1,33.22,444.333]"
        input_names = 'data,info'
        result = get_mean_scale_dictionary(parse_tuple_pairs(mean_values), parse_tuple_pairs(scale_values), input_names)
        exp_res = {
            'data': {
                'mean': np.array([1.1, 22.22, 333.333]),
                'scale': np.array([1.1, 22.22, 333.333])
            },
            'info': {
                'mean': np.array([2.1, 33.22, 444.333]),
                'scale': np.array([2.1, 33.22, 444.333])
            }
        }
        for input in exp_res.keys():
            for key in exp_res[input].keys():
                if type(exp_res[input][key]) is np.ndarray:
                    np.array_equal(exp_res[input][key], result[input][key])
                else:
                    self.assertEqual(exp_res[input][key], result[input][key])

    def test_multi_mean_scale_arrays_no_input(self):
        mean_values = "(1.1,22.22,333.333),(2.1,33.22,444.333)"
        scale_values = "[1.1,22.22,333.333],[2.1,33.22,444.333]"
        result = get_mean_scale_dictionary(parse_tuple_pairs(mean_values), parse_tuple_pairs(scale_values), None)
        exp_res = [
            [
                np.array([1.1, 22.22, 333.333]),  # mean
                np.array([1.1, 22.22, 333.333])  # scale
            ],
            [
                np.array([2.1, 33.22, 444.333]),  # mean
                np.array([2.1, 33.22, 444.333])  # scale
            ]
        ]
        for i in range(0, len(exp_res)):
            for j in range(0, len(exp_res[i])):
                np.array_equal(exp_res[i][j], result[i][j])


class TestSingleTupleParsing(unittest.TestCase):
    def test_get_values_ideal(self):
        values = "(1.11, 22.22, 333.333)"
        result = get_tuple_values(values)
        exp_res = ['1.11, 22.22, 333.333']
        self.assertEqual(exp_res, result)

    def test_get_values_ideal_spaces(self):
        values = "(1    , 22 ,333)"
        result = get_tuple_values(values)
        exp_res = ['1    , 22 ,333']
        self.assertEqual(exp_res, result)

    def test_get_values_ideal_square(self):
        values = "[1,22,333]"
        result = get_tuple_values(values)
        exp_res = ['1,22,333']
        self.assertEqual(exp_res, result)

    def test_get_values_ideal_square_spaces(self):
        values = "[1    , 22 ,333]"
        result = get_tuple_values(values)
        exp_res = ['1    , 22 ,333']
        self.assertEqual(exp_res, result)

    def test_get_neg_values_ideal(self):
        values = "(-1,-22,-333)"
        result = get_tuple_values(values)
        exp_res = ['-1,-22,-333']
        self.assertEqual(exp_res, result)

    def test_get_neg_values_minus(self):
        values = "(-1,--22,-3-33)"
        self.assertRaises(Error, get_tuple_values, values)

    def test_get_values_unbalanced(self):
        values = "(1,22,333]"
        self.assertRaises(Error, get_tuple_values, values)

    def test_get_values_unbalanced2(self):
        values = "[1,22,333)"
        self.assertRaises(Error, get_tuple_values, values)

    def test_get_values_exactly_3(self):
        values = "[1,22,333,22]"
        self.assertRaises(Error, get_tuple_values, values)

    def test_get_values_exactly_3_1(self):
        values = "[1,22]"
        self.assertRaises(Error, get_tuple_values, values)

    def test_get_values_empty(self):
        values = ""
        self.assertRaises(Error, get_tuple_values, values)

    def test_get_values_empty_tuple(self):
        values = ()
        result = get_tuple_values(values)
        exp_res = ()
        self.assertEqual(exp_res, result)


class TestShapesParsing(unittest.TestCase):
    def test_get_shapes_several_inputs_several_shapes(self):
        argv_input = "inp1,inp2"
        input_shapes = "(1,22,333,123), (-1,45,7,1)"
        result = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([1, 22, 333, 123]), 'inp2': np.array([-1, 45, 7, 1])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            np.testing.assert_array_equal(result[i], exp_res[i])

    def test_get_shapes_several_inputs_several_shapes_not_equal(self):
        argv_input = "inp1,inp2,inp3"
        input_shapes = "(1,22,333,123), (-1,45,7,1)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_several_shapes_one_input(self):
        argv_input = "inp1"
        input_shapes = "(1,22,333,123), (-1,45,7,1), (-1,456,7,1)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_several_shapes_no_input(self):
        argv_input = ""
        input_shapes = "(1,22,333,123), (-1,45,7,1), (-1,456,7,1)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_one_shape(self):
        argv_input = "inp1"
        input_shapes = "(1,22,333,123)"
        result = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([1, 22, 333, 123])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            np.testing.assert_array_equal(result[i], exp_res[i])

    def test_get_shapes_no_input_no_shape(self):
        argv_input = ""
        input_shapes = ""
        result = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = np.array([None])
        np.testing.assert_array_equal(result, exp_res)

    def test_get_shapes_no_input_one_shape(self):
        argv_input = ""
        input_shapes = "(12,4,1)"
        result = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = np.array([12, 4, 1])
        np.testing.assert_array_equal(result, exp_res)

    def test_get_shapes_no_input_one_shape2(self):
        argv_input = ""
        input_shapes = "[12,4,1]"
        result = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = np.array([12, 4, 1])
        np.testing.assert_array_equal(result, exp_res)

    def test_get_shapes_no_input_two_shapes(self):
        argv_input = ""
        input_shapes = "(12,4,1),(5,4,3)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_no_shape(self):
        argv_input = "inp1"
        input_shapes = ""
        result = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([None])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            np.testing.assert_array_equal(result[i], exp_res[i])

    def test_get_shapes_one_input_wrong_shape8(self):
        argv_input = "inp1"
        input_shapes = "[2,4,1)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape9(self):
        argv_input = "inp1"
        input_shapes = "(2,4,1]"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape10(self):
        argv_input = "inp1"
        input_shapes = "(2,,,4,1]"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape2(self):
        argv_input = "inp1"
        input_shapes = "(2,4,1"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape3(self):
        argv_input = "inp1"
        input_shapes = "2,4,1"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape4(self):
        argv_input = "inp1"
        input_shapes = "2;4;1"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape5(self):
        argv_input = "inp1"
        input_shapes = "2,         4,1"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape6(self):
        argv_input = "inp1"
        input_shapes = "(2,         4,1),[4,6,8]"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_wrong_shape7(self):
        argv_input = "inp1"
        input_shapes = "[2,4,1],(4,6,8)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_several_shapes(self):
        argv_input = "inp1"
        input_shapes = "(2,4,1),(4,6,8)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_first_neg_shape1(self):
        argv_input = "inp1,inp2"
        input_shapes = "(-1,4,1),(4,6,8)"
        result = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([-1, 4, 1]), 'inp2': np.array([4, 6, 8])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            np.testing.assert_array_equal(result[i], exp_res[i])

    def test_get_shapes_one_input_first_neg_shape_not_one(self):
        argv_input = "inp1"
        input_shapes = "(-12,4,1),(4,6,8)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_any_neg_shape(self):
        argv_input = "inp1, inp2"
        input_shapes = "(12,4,1),(4,-6,8)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)


class TestModelNameParsing(unittest.TestCase):
    def test_model_name_ideal(self):
        model_name = '/home/models/mymodel.caffemodel'
        res = get_model_name(model_name)
        exp_res = 'mymodel'
        self.assertEqual(exp_res, res)

    def test_model_name_no_name(self):
        model_name = '/home/models/.caffemodel'
        res = get_model_name(model_name)
        exp_res = 'model'
        self.assertEqual(exp_res, res)

    def test_model_name_no_ext(self):
        model_name = '/home/models/caffemodel'
        res = get_model_name(model_name)
        exp_res = 'caffemodel'
        self.assertEqual(exp_res, res)

    def test_model_name_no_name_no_path(self):
        model_name = '.caffemodel'
        res = get_model_name(model_name)
        exp_res = 'model'
        self.assertEqual(exp_res, res)

    @patch("mo.utils.cli_parser.os")
    def test_model_name_win(self, old_os):
        old_os.path.basename.return_value = "caffemodel"
        old_os.path.splitext.return_value = ("caffemodel", "")
        model_name = r'\home\models\caffemodel'
        res = get_model_name(model_name)

        exp_res = 'caffemodel'
        self.assertEqual(exp_res, res)

    def test_model_name_dots(self):
        model_name = r'/home/models/squeezenet_v1.1.caffemodel'
        res = get_model_name(model_name)
        exp_res = 'squeezenet_v1.1'
        self.assertEqual(exp_res, res)


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
            os.removedirs(__class__.NOT_WRITABLE_DIR)
        if os.path.exists(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR)):
            os.removedirs(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR))
        if os.path.exists(__class__.EXISTING_FILE):
            os.remove(__class__.EXISTING_FILE)

    def test_single_writable_dir(self):
        self.assertEqual(__class__.WRITABLE_DIR, writable_dir(__class__.WRITABLE_DIR))

    def test_single_non_writable_dir(self):
        with self.assertRaises(Error) as cm:
            writable_dir(__class__.NOT_WRITABLE_DIR)

    def test_single_non_writable_sub_dir(self):
        with self.assertRaises(Error) as cm:
            writable_dir(__class__.NOT_WRITABLE_SUB_DIR)

    def test_multiple_writable_dirs(self):
        dirs_str = ','.join([__class__.WRITABLE_DIR, __class__.WRITABLE_NON_EXISTING_DIR])
        self.assertEqual(dirs_str, writable_dir(dirs_str))

    def test_single_writable_non_existing_dir(self):
        self.assertEqual(__class__.WRITABLE_NON_EXISTING_DIR, writable_dir(__class__.WRITABLE_NON_EXISTING_DIR))

    def test_readable_dirs(self):
        dirs_str = ','.join([__class__.WRITABLE_DIR, __class__.READABLE_DIR])
        self.assertEqual(dirs_str, readable_dirs(dirs_str))

    def test_not_readable_dirs(self):
        dirs_str = ','.join([__class__.WRITABLE_DIR, __class__.WRITABLE_NON_EXISTING_DIR])
        with self.assertRaises(Error) as cm:
            readable_dirs(dirs_str)

    def test_readable_file(self):
        self.assertEqual(__class__.EXISTING_FILE, readable_file(__class__.EXISTING_FILE))

    def test_non_readable_file(self):
        with self.assertRaises(Error) as cm:
            readable_file(__class__.NOT_EXISTING_FILE)
