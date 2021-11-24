# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import unittest
from argparse import Namespace

from mo.utils.error import Error

import numpy as np


try:
    # pylint: disable=no-name-in-module,import-error
    from mo.moc_frontend.preprocessing import apply_preprocessing

    # pylint: disable=no-name-in-module,import-error
    import openvino as ov
    import openvino.opset8 as ops
    from openvino import Function, PartialShape

except Exception:
    print("No OpenVINO API available,"
          "ensure to set correct PYTHONPATH when running these tests")
    raise


def create_function2(shape1=[2, 2], shape2=[2, 2]):
    input1 = ops.parameter(shape1, dtype=np.float32, name="input1")
    relu1 = ops.relu(input1)
    res1 = ops.result(relu1, "res1")
    input2 = ops.parameter(shape2, dtype=np.float32, name="input2")
    relu2 = ops.relu(input2)
    res2 = ops.result(relu2, "res2")
    function = Function(results=[res1, res2], parameters=[input1, input2], name="TestFunction")
    return function


def process_function(ov_function: Function, argv: Namespace):
    apply_preprocessing(ov_function=ov_function,
                        argv=argv)


class TestPreprocessingMOC(unittest.TestCase):
    def setUp(self):
        pass

    def check_scale_constant(self, node, expected, shape=None):
        const_node = node.input(1).get_source_output().get_node()
        self.assertEqual(const_node.get_type_name(), 'Constant')
        if node.get_type_name() == 'Divide':
            self.assertTrue(np.allclose(const_node.get_vector(), expected))
        else:
            self.assertTrue(np.allclose(const_node.get_vector(), 1. / expected))
        if shape:
            assert const_node.shape == PartialShape(shape)

    def check_mean_constant(self, node, expected, shape=None):
        const_node = node.input(1).get_source_output().get_node()
        self.assertEqual(const_node.get_type_name(), 'Constant')
        if node.get_type_name() == 'Subtract':
            self.assertTrue(np.allclose(const_node.get_vector(), expected))
        else:
            self.assertTrue(np.allclose(const_node.get_vector(), -expected.toList()))
        if shape:
            self.assertEqual(const_node.shape, PartialShape(shape))

    def test_scale_single_value(self):
        argv = Namespace(mean_scale_values=None, scale=2.0)
        function = create_function2()
        process_function(ov_function=function, argv=argv)

        for param in function.get_parameters():
            op_node = list(param.output(0).get_target_inputs())[0].get_node()
            self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
            self.check_scale_constant(op_node, [2.0])

    def test_scale_vector(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([4.]), 'mean': None}}, scale=None)
        function = create_function2()
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, [4.0], shape=None)
        # Verify that input2 is not affected
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_scale_vector3(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([2., 4., 8.]), 'mean': None}}, scale=None)
        function = create_function2(shape1=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        # Verify that input2 is not affected
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_mean_single(self):
        argv = Namespace(mean_scale_values={'input1': {'mean': np.array([4.]), 'scale': None}}, scale=None)
        function = create_function2()
        apply_preprocessing(ov_function=function,
                            argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, [4.0], shape=None)
        # Verify that input2 is not affected
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_mean_vector3(self):
        argv = Namespace(mean_scale_values={'input2': {'mean': np.array([2., 4., 8.]), 'scale': None}}, scale=None)
        function = create_function2(shape2=[1, 3, 224, 224])
        apply_preprocessing(ngraph_function=function,
                            argv=argv)
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        # Verify that input1 is not affected
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_mean_scale(self):
        argv = Namespace(mean_scale_values={'input2': {'mean': np.array([1., 2., 3.]),
                                                       'scale': np.array([2., 4., 8.])}}, scale=None)
        function = create_function2(shape2=[1, 3, 224, 224])
        apply_preprocessing(ov_function=function,
                            argv=argv)
        # Verify that first is 'subtract mean', then 'scale'
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1., 2., 3.], shape=[1, 3, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        # Verify that input1 is not affected
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_no_param_name(self):
        argv = Namespace(mean_scale_values=list(np.array([(np.array([1., 2., 3.]), np.array([2., 4., 6.])),
                                                          (np.array([7., 8.]), None)],
                                                         dtype='object')), scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 2, 224, 224])
        apply_preprocessing(ov_function=function,
                            argv=argv)

        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1., 2., 3.], shape=[1, 3, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 6.], shape=[1, 3, 1, 1])

        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[7., 8.], shape=[1, 2, 1, 1])

    def test_no_param_name_single_value(self):
        argv = Namespace(mean_scale_values=list(np.array([(np.array([1.]), None),
                                                          (np.array([2., 3.]), np.array([4.]))],
                                                         dtype='object')), scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 2, 224, 224])
        apply_preprocessing(ov_function=function,
                            argv=argv)

        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1.], shape=None)

        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 3.], shape=[1, 2, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[4.], shape=None)

    # Two inputs, but 'mean_scale_value' has only one array
    def test_error_no_param_name_number_not_match(self):
        argv = Namespace(mean_scale_values=[(np.array([2., 3.]), np.array([4.]))], scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 2, 224, 224])
        with self.assertRaisesRegex(Error, '.*question.*61.*'):
            apply_preprocessing(ov_function=function,
                                argv=argv)

    def test_error_no_node_name_found(self):
        argv = Namespace(mean_scale_values={'not_found': {'scale': np.array([1.]), 'mean': np.array([1.])}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 2, 224, 224])
        with self.assertRaisesRegex(Error, '.*question.*83.*'):
            apply_preprocessing(ov_function=function,
                                argv=argv)

    def test_error_dimension_mismatch(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3., 4.]), 'mean': None}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 224, 224])
        with self.assertRaises(Exception):
            apply_preprocessing(ov_function=function,
                                argv=argv)
