# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

from mo.utils.cli_parser import get_placeholder_shapes, get_tuple_values, get_mean_scale_dictionary, get_model_name, \
    parse_tuple_pairs, check_positive, writable_dir, readable_dirs, \
    readable_file, get_freeze_placeholder_values, parse_transform, check_available_transforms
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
            npt.assert_array_equal(result[el], exp_res[el])

    def test_tuple_parser_name_digits_only(self):
        tuple_values = "0448(1.1,22.22,333.333),0449[2.2,33.33,444.444]"
        result = parse_tuple_pairs(tuple_values)
        exp_res = {
            '0448': np.array([1.1, 22.22, 333.333]),
            '0449': np.array([2.2, 33.33, 444.444])
        }
        for el in exp_res.keys():
            npt.assert_array_equal(result[el], exp_res[el])

    def test_tuple_parser_same_values(self):
        tuple_values = "data(1.1,22.22,333.333),info[1.1,22.22,333.333]"
        result = parse_tuple_pairs(tuple_values)
        exp_res = {
            'data': np.array([1.1, 22.22, 333.333]),
            'info': np.array([1.1, 22.22, 333.333])
        }
        for el in exp_res.keys():
            npt.assert_array_equal(result[el], exp_res[el])

    def test_tuple_parser_no_inputs(self):
        tuple_values = "(1.1,22.22,333.333),[2.2,33.33,444.444]"
        result = parse_tuple_pairs(tuple_values)
        exp_res = [np.array([1.1, 22.22, 333.333]),
                   np.array([2.2, 33.33, 444.444])]
        for i in range(0, len(exp_res)):
            npt.assert_array_equal(result[i], exp_res[i])

    def test_tuple_parser_error_mixed_with_and_without_name(self):
        tuple_values = "(1.1,22.22,333.333),data[2.2,33.33,444.444]"
        self.assertRaises(Error, parse_tuple_pairs, tuple_values)

    def test_tuple_parser_error_mixed_with_and_without_name_1(self):
        tuple_values = "data(1.1,22.22,333.333),[2.2,33.33,444.444]"
        self.assertRaises(Error, parse_tuple_pairs, tuple_values)

    def test_tuple_parser_error_mixed_with_and_without_name_digits(self):
        tuple_values = "(0.1,22.22,333.333),0448[2.2,33.33,444.444]"
        self.assertRaises(Error, parse_tuple_pairs, tuple_values)

    def test_tuple_parser_error_mixed_with_and_without_name_digits_1(self):
        tuple_values = "447(1.1,22.22,333.333),[2.2,33.33,444.444]"
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[i][j], result[i][j])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[i][j], result[i][j])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                    npt.assert_array_equal(exp_res[input][key], result[input][key])
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
                npt.assert_array_equal(exp_res[i][j], result[i][j])

    def test_scale_do_not_match_input(self):
        scale_values = parse_tuple_pairs("input_not_present(255),input2(255)")
        mean_values = parse_tuple_pairs("input1(255),input2(255)")
        self.assertRaises(Error, get_mean_scale_dictionary, mean_values, scale_values, "input1,input2")

    def test_mean_do_not_match_input(self):
        scale_values = parse_tuple_pairs("input1(255),input2(255)")
        mean_values = parse_tuple_pairs("input_not_present(255),input2(255)")
        self.assertRaises(Error, get_mean_scale_dictionary, mean_values, scale_values, "input1,input2")

    def test_values_match_input_name(self):
        # to be sure that we correctly processes complex names
        res_values = parse_tuple_pairs("input255(255),input255.0(255.0),multi-dotted.input.3.(255,128,64)")
        exp_res = {'input255': np.array([255.0]),
                   'input255.0': np.array([255.0]),
                   'multi-dotted.input.3.': np.array([255., 128., 64.])}
        self.assertEqual(len(exp_res), len(res_values))
        for i, j in zip(exp_res, res_values):
            self.assertEqual(i, j)
            npt.assert_array_equal(exp_res[i], res_values[j])

    def test_input_without_values(self):
        self.assertRaises(Error, parse_tuple_pairs, "input1,input2")

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
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([1, 22, 333, 123]), 'inp2': np.array([-1, 45, 7, 1])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])

    def test_get_shapes_several_inputs_several_shapes2(self):
        # shapes specified using --input command line parameter and no values
        argv_input = "inp1[1 22 333 123],inp2[-1 45 7 1]"
        result, _ = get_placeholder_shapes(argv_input, None)
        exp_res = {'inp1': np.array([1, 22, 333, 123]), 'inp2': np.array([-1, 45, 7, 1])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input, None)
        placeholder_values_ref = {}
        input_node_names_ref = "inp1,inp2"
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            npt.assert_array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_shapes3(self):
        # shapes and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3[5]->[1.0 1.0 2.0 3.0 5.0]"
        result, _ = get_placeholder_shapes(argv_input, None)
        exp_res = {'inp1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), 'inp3': np.array([5])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input, None)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': np.array(['1.0', '1.0', '2.0', '3.0', '5.0'])}
        input_node_names_ref = "inp1,inp2,inp3"
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            npt.assert_array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_shapes4(self):
        # shapes specified using --input_shape and values for freezing using --input command line parameter
        argv_input = "inp1->[1.0 2.0 3.0],inp2,inp3->[1.0 1.0 2.0 3.0 5.0]"
        input_shapes = "(3,1), (3,2,3), (5)"
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), 'inp3': np.array([5])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input, None)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': np.array(['1.0', '1.0', '2.0', '3.0', '5.0'])}
        input_node_names_ref = "inp1,inp2,inp3"
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            npt.assert_array_equal(placeholder_values_res[i], placeholder_values_ref[i])
        self.assertEqual(input_node_names_ref, input_node_names_res)

    def test_get_shapes_several_inputs_several_shapes5(self):
        # some values for freezing specified using --freeze_placeholder_with_value
        argv_input = "inp1->[1.0 2.0 3.0],inp2,inp3->[1.0 1.0 2.0 3.0 5.0]"
        input_shapes = "(3,1), (3,2,3), (5)"
        argv_freeze_placeholder_with_value = "inp2->[5.0 7.0 3.0],inp4->[100.0 200.0]"

        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), 'inp3': np.array([5])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input, argv_freeze_placeholder_with_value)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': np.array(['1.0', '1.0', '2.0', '3.0', '5.0'],),
                                  'inp2': np.array(['5.0', '7.0', '3.0']), 'inp4': np.array(['100.0', '200.0'])}
        input_node_names_ref = "inp1,inp2,inp3"
        self.assertEqual(sorted(list(placeholder_values_res.keys())), sorted(list(placeholder_values_ref.keys())))
        for i in placeholder_values_ref.keys():
            npt.assert_array_equal(placeholder_values_res[i], placeholder_values_ref[i])
        self.assertEqual(input_node_names_ref, input_node_names_res)

    def test_get_shapes_several_inputs_several_shapes6(self):
        # 0D value for freezing specified using --input command line parameter without shape
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3->False"
        result, _ = get_placeholder_shapes(argv_input, None)
        exp_res = {'inp1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), 'inp3': np.array(False).shape}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input, None)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': False}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            npt.assert_array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_several_inputs_several_shapes7(self):
        # 0D shape and value for freezing specified using --input command line parameter
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3[]->True"
        result, _ = get_placeholder_shapes(argv_input, None)
        exp_res = {'inp1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), 'inp3': np.array(False).shape}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])
        placeholder_values_res, input_node_names_res = get_freeze_placeholder_values(argv_input, None)
        placeholder_values_ref = {'inp1': np.array(['1.0', '2.0', '3.0']), 'inp3': True}
        self.assertEqual(list(placeholder_values_res.keys()), list(placeholder_values_ref.keys()))
        for i in placeholder_values_ref.keys():
            npt.assert_array_equal(placeholder_values_res[i], placeholder_values_ref[i])

    def test_get_shapes_and_data_types1(self):
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{i32},inp3[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        result_shapes, result_data_types = get_placeholder_shapes(argv_input, "")
        ref_result_shapes = {'inp1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), 'inp3': np.array([5])}
        ref_result_data_types = {'inp2': np.int32, 'inp3': np.float32}
        self.assertEqual(list(ref_result_shapes.keys()), list(result_shapes.keys()))
        for i in ref_result_shapes.keys():
            npt.assert_array_equal(result_shapes[i], ref_result_shapes[i])
        self.assertEqual(list(ref_result_data_types.keys()), list(result_data_types.keys()))
        for i in ref_result_data_types.keys():
            np.testing.assert_equal(result_data_types[i], ref_result_data_types[i])

    def test_get_shapes_and_data_types_with_input_ports(self):
        argv_input = "1:inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{i32},0:inp3[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        result_shapes, result_data_types = get_placeholder_shapes(argv_input, "")
        ref_result_shapes = {'1:inp1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), '0:inp3': np.array([5])}
        ref_result_data_types = {'inp2': np.int32, '0:inp3': np.float32}
        self.assertEqual(list(ref_result_shapes.keys()), list(result_shapes.keys()))
        for i in ref_result_shapes.keys():
            npt.assert_array_equal(result_shapes[i], ref_result_shapes[i])
        self.assertEqual(list(ref_result_data_types.keys()), list(result_data_types.keys()))
        for i in ref_result_data_types.keys():
            np.testing.assert_equal(result_data_types[i], ref_result_data_types[i])

    def test_get_shapes_and_data_types_with_output_ports(self):
        argv_input = "inp1:1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{i32},inp3:4[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        result_shapes, result_data_types = get_placeholder_shapes(argv_input, "")
        ref_result_shapes = {'inp1:1': np.array([3, 1]), 'inp2': np.array([3, 2, 3]), 'inp3:4': np.array([5])}
        ref_result_data_types = {'inp2': np.int32, 'inp3:4': np.float32}
        self.assertEqual(list(ref_result_shapes.keys()), list(result_shapes.keys()))
        for i in ref_result_shapes.keys():
            npt.assert_array_equal(result_shapes[i], ref_result_shapes[i])
        self.assertEqual(list(ref_result_data_types.keys()), list(result_data_types.keys()))
        for i in ref_result_data_types.keys():
            np.testing.assert_equal(result_data_types[i], ref_result_data_types[i])

    def test_get_shapes_and_data_types_shape_only(self):
        argv_input = "placeholder1[3 1],placeholder2,placeholder3"
        result_shapes, result_data_types = get_placeholder_shapes(argv_input, "")
        ref_result_shapes = {'placeholder1': np.array([3, 1]), 'placeholder2': None,
                             'placeholder3': None}
        ref_result_data_types = {}
        self.assertEqual(list(ref_result_shapes.keys()), list(result_shapes.keys()))
        for i in ref_result_shapes.keys():
            npt.assert_array_equal(result_shapes[i], ref_result_shapes[i])
        self.assertEqual(list(ref_result_data_types.keys()), list(result_data_types.keys()))
        for i in ref_result_data_types.keys():
            np.testing.assert_equal(result_data_types[i], ref_result_data_types[i])

    def test_get_shapes_and_data_types_shape_with_ports_only(self):
        argv_input = "placeholder1:4[3 1],placeholder2,2:placeholder3"
        result_shapes, result_data_types = get_placeholder_shapes(argv_input, "")
        ref_result_shapes = {'placeholder1:4': np.array([3, 1]), 'placeholder2': None,
                             '2:placeholder3': None}
        ref_result_data_types = {}
        self.assertEqual(list(ref_result_shapes.keys()), list(result_shapes.keys()))
        for i in ref_result_shapes.keys():
            npt.assert_array_equal(result_shapes[i], ref_result_shapes[i])
        self.assertEqual(list(ref_result_data_types.keys()), list(result_data_types.keys()))
        for i in ref_result_data_types.keys():
            np.testing.assert_equal(result_data_types[i], ref_result_data_types[i])

    def test_get_shapes_and_data_types_when_no_freeze_value(self):
        argv_input = "placeholder1{i32}[3 1],placeholder2,placeholder3{i32}"
        result_shapes, result_data_types = get_placeholder_shapes(argv_input, "")
        ref_result_shapes = {'placeholder1': np.array([3, 1]), 'placeholder2': None,
                             'placeholder3': None}
        ref_result_data_types = {'placeholder1': np.int32, 'placeholder3': np.int32}
        self.assertEqual(list(ref_result_shapes.keys()), list(result_shapes.keys()))
        for i in ref_result_shapes.keys():
            npt.assert_array_equal(result_shapes[i], ref_result_shapes[i])
        self.assertEqual(list(ref_result_data_types.keys()), list(result_data_types.keys()))
        for i in ref_result_data_types.keys():
            np.testing.assert_equal(result_data_types[i], ref_result_data_types[i])

    def test_wrong_data_types(self):
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3]{abracadabra},inp3[5]{f32}->[1.0 1.0 2.0 3.0 5.0]"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, "")

    def test_shapes_specified_using_both_params(self):
        # shapes specified using both command line parameter --input and --input_shape
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3[5]->[1.0 1.0 2.0 3.0 5.0]"
        input_shapes = "(3,1), (3,2,3), (5)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_shape_and_value_shape_mismatch(self):
        # size of value tensor does not correspond to specified shape for the third node
        argv_input = "inp1[3 1]->[1.0 2.0 3.0],inp2[3 2 3],inp3[5 3]->[2.0 3.0 5.0]"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, None)

    def test_wrong_data_for_input_cmd_param(self):
        # test that wrongly formatted data specified in --input is handled properly
        argv_input = "abc->[1.0"
        self.assertRaises(Error, get_freeze_placeholder_values, argv_input, None)
        argv_input = "def[2 2]->[1.0 2.0 3.0 4.0],abc->1.0 34]"
        self.assertRaises(Error, get_freeze_placeholder_values, argv_input, None)

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
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([1, 22, 333, 123])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])

    def test_get_shapes_no_input_no_shape(self):
        argv_input = ""
        input_shapes = ""
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = np.array([None])
        npt.assert_array_equal(result, exp_res)

    def test_get_shapes_no_input_one_shape(self):
        argv_input = ""
        input_shapes = "(12,4,1)"
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = np.array([12, 4, 1])
        npt.assert_array_equal(result, exp_res)

    def test_get_shapes_no_input_one_shape2(self):
        argv_input = ""
        input_shapes = "[12,4,1]"
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = np.array([12, 4, 1])
        npt.assert_array_equal(result, exp_res)

    def test_get_shapes_no_input_two_shapes(self):
        argv_input = ""
        input_shapes = "(12,4,1),(5,4,3)"
        self.assertRaises(Error, get_placeholder_shapes, argv_input, input_shapes)

    def test_get_shapes_one_input_no_shape(self):
        argv_input = "inp1"
        input_shapes = ""
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([None])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])

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
        result, _ = get_placeholder_shapes(argv_input, input_shapes)
        exp_res = {'inp1': np.array([-1, 4, 1]), 'inp2': np.array([4, 6, 8])}
        self.assertEqual(list(exp_res.keys()), list(result.keys()))
        for i in exp_res.keys():
            npt.assert_array_equal(result[i], exp_res[i])

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
            shutil.rmtree(__class__.NOT_WRITABLE_DIR, ignore_errors=True)
        if os.path.exists(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR)):
            shutil.rmtree(os.path.dirname(__class__.NOT_WRITABLE_SUB_DIR), ignore_errors=True)
        if os.path.exists(__class__.EXISTING_FILE):
            os.remove(__class__.EXISTING_FILE)

    def test_single_writable_dir(self):
        self.assertEqual(__class__.WRITABLE_DIR, writable_dir(__class__.WRITABLE_DIR))

    @unittest.skipIf(sys.platform.startswith("win"), "chmod() on Windows do nor support not writable dir")
    def test_single_non_writable_dir(self):
        with self.assertRaises(Error) as cm:
            writable_dir(__class__.NOT_WRITABLE_DIR)

    @unittest.skipIf(sys.platform.startswith("win"), "chmod() on Windows do nor support not writable dir")
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


class TransformChecker(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(parse_transform(""), [])

    def test_single_pass(self):
        self.assertEqual(parse_transform("LowLatency2"), [("LowLatency2", {})])

    def test_single_pass_with_args(self):
        self.assertEqual(parse_transform("LowLatency2[use_const_initializer=True]"),
                         [("LowLatency2", {"use_const_initializer": True})])

    def test_single_pass_with_multiple_args(self):
        self.assertEqual(parse_transform("LowLatency2[use_const_initializer=True;dummy_attr=3.14]"),
                         [("LowLatency2", {"use_const_initializer": True, "dummy_attr": 3.14})])

    def test_multiple_passes_with_args(self):
        self.assertEqual(parse_transform("LowLatency2[use_const_initializer=True],DummyPass[type=ReLU]"),
                         [("LowLatency2", {"use_const_initializer": True}),
                          ("DummyPass", {"type": "ReLU"})])

    def test_multiple_passes_with_args2(self):
        self.assertEqual(parse_transform("LowLatency2[use_const_initializer=True,False],DummyPass1,"
                                         "DummyPass2[types=ReLU,PReLU;values=1,2,3]"),
                         [("LowLatency2",  {"use_const_initializer": [True, False]}),
                          ("DummyPass1",  {}),
                          ("DummyPass2",  {"types": ["ReLU", "PReLU"], "values": [1,2,3]})])

    def test_multiple_passes_no_args(self):
        self.assertEqual(parse_transform("DummyPass,LowLatency22"),
                         [("DummyPass", {}), ("LowLatency22", {})])

    def test_single_pass_neg(self):
        self.assertRaises(Error, parse_transform, "LowLatency2!")

    def test_multiple_passes_neg(self):
        self.assertRaises(Error, parse_transform, "LowLatency2;DummyPass")

    def test_single_pass_with_args_neg1(self):
        self.assertRaises(Error, parse_transform, "LowLatency2[=2]")

    def test_single_pass_with_args_neg2(self):
        self.assertRaises(Error, parse_transform, "LowLatency2[key=]")

    def test_single_pass_with_args_neg3(self):
        self.assertRaises(Error, parse_transform, "LowLatency2[]")

    def test_single_pass_with_args_neg4(self):
        self.assertRaises(Error, parse_transform, "LowLatency2[key=value;]")

    def test_single_pass_with_args_neg5(self):
        self.assertRaises(Error, parse_transform, "LowLatency2[value]")

    def test_single_pass_with_args_neg6(self):
        self.assertRaises(Error, parse_transform, "LowLatency2[key=value")

    @patch("mo.back.offline_transformations.get_available_transformations")
    def test_check_low_latency_is_available(self, available_transformations):
        available_transformations.return_value = {"LowLatency2": None}
        try:
            check_available_transforms([("LowLatency2", "")], True)
        except Error as e:
            self.assertTrue(False, "Exception \"{}\" is unexpected".format(e))

    @patch("mo.back.offline_transformations.get_available_transformations")
    def test_check_dummy_pass_is_available(self, available_transformations):
        available_transformations.return_value = {"LowLatency2": None}
        self.assertRaises(Error, check_available_transforms, [("DummyPass", "")], True)
