# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from argparse import Namespace

import numpy as np
from openvino.tools.mo.utils.error import Error
from unit_tests.mo.unit_test_with_mocked_telemetry import UnitTestWithMockedTelemetry

try:
    # pylint: disable=no-name-in-module,import-error
    from openvino.tools.mo.back.preprocessing import apply_preprocessing

    # pylint: disable=no-name-in-module,import-error
    import openvino.runtime.opset8 as ops
    from openvino.runtime import Model, Layout, PartialShape

except Exception:
    print("No OpenVINO API available,"
          "ensure to set correct PYTHONPATH when running these tests")
    raise


def create_function3(shape1=[2, 2]):
    input1 = ops.parameter(shape1, dtype=np.float32, name="input1")
    input1.get_output_tensor(0).set_names({'a_input', 'b_input', 'c_input'})
    relu1 = ops.relu(input1)
    res1 = ops.result(relu1, "res")
    res1.get_output_tensor(0).set_names({'res'})
    function = Model(results=[res1], parameters=[input1], name="TestFunction")
    return function


def create_function2(shape1=[2, 2], shape2=[2, 2], dtype1=np.float32, dtype2=np.float32):
    input1 = ops.parameter(shape1, dtype=dtype1, name="input1")
    input1.get_output_tensor(0).set_names({'input1', 'input1a'})
    relu1 = ops.relu(input1)
    res1 = ops.result(relu1, "res1")
    res1.get_output_tensor(0).set_names({'res1', 'res1a'})
    input2 = ops.parameter(shape2, dtype=dtype2, name="input2")
    input2.get_output_tensor(0).set_names({'input2', 'input2a'})
    relu2 = ops.relu(input2)
    res2 = ops.result(relu2, "res2")
    res2.get_output_tensor(0).set_names({'res2', 'res2a'})
    function = Model(results=[res1, res2], parameters=[input1, input2], name="TestFunction")
    return function


def create_function1(shape1=[2, 2]):
    input1 = ops.parameter(shape1, dtype=np.float32, name="input1")
    input1.get_output_tensor(0).set_names({'input1a', 'input1b'})
    relu1 = ops.relu(input1)
    res1 = ops.result(relu1, "res1")
    res1.get_output_tensor(0).set_names({'res1', 'res1a'})
    function = Model(results=[res1], parameters=[input1], name="TestFunction")
    return function


def process_function(ov_function: Model, argv: Namespace):
    apply_preprocessing(ov_function=ov_function, argv=argv)


class TestPreprocessingMOC(UnitTestWithMockedTelemetry):
    def setUp(self):
        super(TestPreprocessingMOC, self).setUp()
        pass

    def check_constant(self, const_node, expected, shape=None):
        self.assertEqual(const_node.get_type_name(), 'Constant')
        self.assertTrue(np.allclose(const_node.get_vector(), expected))
        if shape is not None:
            assert const_node.shape == PartialShape(shape)

    def check_scale_constant(self, node, expected, shape=None):
        const_node = node.input(1).get_source_output().get_node()
        if node.get_type_name() != 'Divide':
            expected = 1. / expected
        self.check_constant(const_node, expected, shape)

    def check_mean_constant(self, node, expected, shape=None):
        const_node = node.input(1).get_source_output().get_node()
        if node.get_type_name() != 'Subtract':
            expected = -expected.toList()
        self.check_constant(const_node, expected, shape)

    def test_scale_single_value(self):
        argv = Namespace(mean_scale_values=None, scale=2.0)
        function = create_function2()
        process_function(ov_function=function, argv=argv)

        for param in function.get_parameters():
            op_node = list(param.output(0).get_target_inputs())[0].get_node()
            self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
            self.check_scale_constant(op_node, [2.0])

    def test_scale_single_value_fp64(self):
        argv = Namespace(mean_scale_values=None, scale=2.0)
        function = create_function2(dtype1=np.float64)
        process_function(ov_function=function, argv=argv)

        for ov_input in function.inputs:
            op_node = list(ov_input.get_target_inputs())[0].get_node()
            self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
            self.check_scale_constant(op_node, [2.0])

    def test_scale_single_value_fp16(self):
        argv = Namespace(mean_scale_values=None, scale=2.0)
        function = create_function2(dtype1=np.float16)
        process_function(ov_function=function, argv=argv)

        for ov_input in function.inputs:
            op_node = list(ov_input.get_target_inputs())[0].get_node()
            self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')

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

        # Verify that guessed layout (?C??) is not appeared in input1
        self.assertEqual(function.get_parameters()[0].layout, Layout())

    def test_scale_vector4_layout(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([2., 4., 8., 9.]), 'mean': None}},
                         layout_values={'input1': {'source_layout': 'nhwc'}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 3, 4])  # Use layout to determine channels dim

        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 8., 9.], shape=[1, 1, 1, 4])

        # Verify that input2 is not affected
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

        # Verify that layout (NHWC) is appeared in input1
        self.assertEqual(function.get_parameters()[0].layout, Layout('nhwc'))

    def test_mean_single(self):
        argv = Namespace(mean_scale_values={'input1': {'mean': np.array([4.]), 'scale': None}}, scale=None)
        function = create_function2()
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, [4.0], shape=None)
        # Verify that input2 is not affected
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_mean_single_fp64(self):
        argv = Namespace(mean_scale_values={'input1': {'mean': np.array([4.]), 'scale': None}}, scale=None)
        function = create_function2(dtype1=np.float64)
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, [4.0], shape=None)
        # Verify that input2 is not affected
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_mean_single_fp16(self):
        argv = Namespace(mean_scale_values={'input1': {'mean': np.array([4.]), 'scale': None}}, scale=None)
        function = create_function2(dtype1=np.float16)
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        # Verify that input2 is not affected
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

    def test_mean_vector3(self):
        argv = Namespace(mean_scale_values={'input2': {'mean': np.array([2., 4., 8.]), 'scale': None}}, scale=None)
        function = create_function2(shape2=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        # Verify that input1 is not affected
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

        # Verify that guessed layout (?C??) is not appeared in input2
        self.assertEqual(function.get_parameters()[1].layout, Layout())

    def test_mean_scale(self):
        argv = Namespace(mean_scale_values={'input2a': {'mean': np.array([1., 2., 3.]),
                                                        'scale': np.array([2., 4., 8.])}},
                         scale=None)
        function = create_function2(shape2=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
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

        # Verify that guessed layout (?C??) is not appeared in input2
        self.assertEqual(function.get_parameters()[1].layout, Layout())

    def test_mean_scale_with_layout(self):
        argv = Namespace(mean_scale_values={'input2a': {'mean': np.array([1., 2., 3., 4.]),
                                                        'scale': np.array([2., 4., 8., 9.])}},
                         scale=None)
        function = create_function2(shape2=[1, 3, 3, 4])
        function.get_parameters()[1].layout = Layout("NHWC")
        process_function(ov_function=function, argv=argv)
        # Verify that first is 'subtract mean', then 'scale'
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1., 2., 3., 4.], shape=[1, 1, 1, 4])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 8., 9.], shape=[1, 1, 1, 4])

        # Verify that input1 is not affected
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

        # Verify that layout presents in function after preprocessing
        self.assertEqual(function.get_parameters()[1].layout, Layout("NHWC"))

    def test_mean_scale_with_layout_dynamic(self):
        argv = Namespace(mean_scale_values={'input2a': {'mean': np.array([1., 2., 3., 4.]),
                                                        'scale': np.array([2., 4., 8., 9.])}},
                         scale=None)
        function = create_function2(shape2=[-1, -1, -1, -1])
        function.get_parameters()[1].layout = Layout("NHWC")
        process_function(ov_function=function, argv=argv)
        # Verify that first is 'subtract mean', then 'scale'
        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1., 2., 3., 4.], shape=[1, 1, 1, 4])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 8., 9.], shape=[1, 1, 1, 4])

        # Verify that input1 is not affected
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

        # Verify that layout presents in function after preprocessing
        self.assertEqual(function.get_parameters()[1].layout, Layout("NHWC"))

    def test_no_param_name(self):
        argv = Namespace(mean_scale_values=list(np.array([(np.array([1., 2., 3.]), np.array([2., 4., 6.])),
                                                          (np.array([7., 8., 9.]), None)],
                                                         dtype='object')), scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 224, 224, 3])
        process_function(ov_function=function, argv=argv)

        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1., 2., 3.], shape=[1, 3, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 6.], shape=[1, 3, 1, 1])

        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[7., 8., 9.], shape=[1, 1, 1, 3])

        # Verify that guessed layouts are not appeared in inputs
        self.assertEqual(function.get_parameters()[0].layout, Layout())
        self.assertEqual(function.get_parameters()[1].layout, Layout())

    def test_no_param_name_single_value(self):
        argv = Namespace(mean_scale_values=list(np.array([(np.array([1.]), None),
                                                          (np.array([2., 3., 4.]), np.array([5.]))],
                                                         dtype='object')), scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 224, 224, 3])
        process_function(ov_function=function, argv=argv)

        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1.], shape=None)

        op_node = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 3., 4.], shape=[1, 1, 1, 3])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[5.], shape=None)

    # Two inputs, but 'mean_scale_value' has only one array
    def test_error_no_param_name_number_not_match(self):
        argv = Namespace(mean_scale_values=[(np.array([2., 3.]), np.array([4.]))], scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 2, 224, 224])
        with self.assertRaisesRegex(Error, '.*question.*61.*'):
            process_function(ov_function=function, argv=argv)

    def test_mean_scale_error_no_node_name_found(self):
        argv = Namespace(mean_scale_values={'not_found': {'scale': np.array([1.]), 'mean': np.array([1.])}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 2, 224, 224])
        with self.assertRaisesRegex(Error, '.*question.*83.*'):
            process_function(ov_function=function, argv=argv)

    def test_layout_error_no_node_name_found(self):
        argv = Namespace(layout_values={'not_found': {'source_layout': 'nhwc'}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 224, 224], shape2=[1, 2, 224, 224])
        with self.assertRaisesRegex(Error, '.*question.*83.*'):
            process_function(ov_function=function, argv=argv)

    def test_error_dimension_mismatch(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3., 4.]), 'mean': None}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 224, 224])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_dimension_not_clear(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3.]), 'mean': None}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 3, 3])  # Not clear to which 3 should scale be applied
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_dimension_mismatch_with_scale(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3., 4.]),
                                                       'mean': np.array([1., 2., 3.])}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 4, 224])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_guess_c_wrong_position_3d(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3.]),
                                                       'mean': np.array([1., 2., 3.])}},
                         scale=None)
        function = create_function2(shape1=[2, 3, 4])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_guess_c_wrong_position_4d(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3.]),
                                                       'mean': np.array([1., 2., 3.])}},
                         scale=None)
        function = create_function2(shape1=[1, 2, 3, 4])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_guess_c_wrong_position_5d(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3.]),
                                                       'mean': np.array([1., 2., 3.])}},
                         scale=None)
        function = create_function2(shape1=[1, 2, 3, 4, 5])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_guess_c_wrong_position_6d(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3.]),
                                                       'mean': np.array([1., 2., 3.])}},
                         scale=None)
        function = create_function2(shape1=[1, 2, 4, 5, 6, 3])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_2_names_to_same_input(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([1., 2., 3.])},
                                            'input1a': {'scale': np.array([1., 2., 3.])}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 224, 224])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_error_2_names_to_same_input_single_value(self):
        argv = Namespace(mean_scale_values={'input1': {'scale': np.array([2.])},
                                            'input1a': {'scale': np.array([3.])}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 224, 224])
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_reverse_input_channels(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None)
        function = create_function2(shape1=[1, 224, 224, 3], shape2=[1, 3, 224, 224])
        process_function(ov_function=function,
                            argv=argv)
        # Verify that some operations are inserted.
        # In future, consider using mock PrePostProcessor to verify that 'reverse_channels' was called
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() != 'Relu')

        # Verify that guessed layouts are not appeared in input1,input2
        self.assertEqual(function.get_parameters()[0].layout, Layout())
        self.assertEqual(function.get_parameters()[1].layout, Layout())

    def test_reverse_input_channels_func_layout(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None)
        function = create_function2(shape1=[1, 3, 3, 3], shape2=[1, 3, 3, 3])
        function.get_parameters()[0].layout = Layout("NCHW")
        function.get_parameters()[1].layout = Layout("NHWC")
        process_function(ov_function=function,
                         argv=argv)
        # Verify that some operations are inserted.
        # In future, consider using mock PrePostProcessor to verify that 'reverse_channels' was called
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() != 'Relu')

        # Verify that guessed layouts are not appeared in input1,input2
        self.assertEqual(function.get_parameters()[0].layout, Layout("NCHW"))
        self.assertEqual(function.get_parameters()[1].layout, Layout("NHWC"))

    def test_reverse_input_channels_layout(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None,
                         layout_values={'input1a': { 'source_layout': 'nhwc' },
                                        'input2a': { 'source_layout': 'nchw' }
                                        })
        function = create_function2(shape1=[1, 224, 224, 4], shape2=[1, 4, 224, 224])
        # no suitable inputs
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_reverse_input_channels_3d(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None,
                         layout_values=None)
        function = create_function2(shape1=[224, 224, 3], shape2=[3, 224, 224])
        process_function(ov_function=function, argv=argv)
        # Verify that reverse_channels are applied.
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() != 'Relu')

    def test_reverse_input_channels_6d(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None,
                         layout_values=None)
        function = create_function2(shape1=[4, 4, 4, 4, 4, 3], shape2=[4, 3, 4, 4, 4, 4])
        # no suitable inputs
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_reverse_input_channels_dynamic(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None,
                         layout_values=None)
        function = create_function2(shape1=[1, -1, 5, 5], shape2=[-1, -1, -1, -1])
        # no suitable inputs
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_reverse_input_channels_dynamic_layout(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None,
                         layout_values={'input1a': {'source_layout': 'nchw'},
                                        'input2a': {'source_layout': 'nhwc'}
                                        })
        function = create_function2(shape1=[1, -1, 5, 5], shape2=[-1, -1, -1, -1])
        process_function(ov_function=function, argv=argv)
        # Verify that reverse_channels are applied.
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() != 'Relu')

    def test_reverse_input_channels_layout_change(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None,
                         layout_values={'input1a': {'source_layout': 'nchw', 'target_layout': 'nhwc'},
                                        'input2a': {'source_layout': 'nhwc', 'target_layout': 'nchw'}
                                        })
        function = create_function2(shape1=[1, 3, 5, 5], shape2=[1, 5, 5, 3])
        process_function(ov_function=function, argv=argv)
        # Verify that reverse_channels are applied.
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() != 'Relu')

    def test_reverse_input_channels_2_channels(self):
        argv = Namespace(reverse_input_channels=True,
                         mean_scale_values=None,
                         scale=None)
        function = create_function2(shape1=[1, 224, 224, 2], shape2=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
        # Verify that some operations are inserted to input2.
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() == 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() != 'Relu')

        # Verify that guessed layouts are not appeared in input1,input2
        self.assertEqual(function.get_parameters()[0].layout, Layout())
        self.assertEqual(function.get_parameters()[1].layout, Layout())

    # When input name for layout is empty for model with one input - it is applied to this input
    def test_scale_vector3_layout_empty_input_name(self):
        argv = Namespace(mean_scale_values=list(np.array([(None, np.array([2., 4., 8.]))],
                                                         dtype='object')),
                         layout_values={'': {'source_layout': 'nchw'}},
                         scale=None)
        function = create_function1(shape1=[1, 3, 3, 3])  # Use layout to determine channels dim

        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        # Verify that layout (nchw) is appeared in input1
        self.assertEqual(function.get_parameters()[0].layout, Layout('nchw'))

    def test_layout_output(self):
        argv = Namespace(mean_scale_values=None,
                         layout_values={
                             'res1': {
                                 'source_layout': 'nchw',
                                 'target_layout': 'nhwc'
                             },
                             'res2a': {
                                 'source_layout': 'ncdhw'
                             }
                         },
                         scale=None)
        function = create_function2(shape1=[1, 3, 3, 3], shape2=[1, 3, 3, 3, 3])

        process_function(ov_function=function, argv=argv)
        op_node = function.get_results()[0].input(0).get_source_output().get_node()
        self.assertEqual(op_node.get_type_name(), 'Transpose')

        self.assertEqual(function.get_results()[0].layout, Layout('nhwc'))
        self.assertEqual(function.get_results()[1].layout, Layout('ncdhw'))

    def test_error_layout_empty_input_name_2_inputs(self):
        argv = Namespace(mean_scale_values=None,
                         layout_values={'': {'source_layout': 'nchw'}},
                         scale=None)
        function = create_function2(shape1=[1, 3, 3, 3])

        # Verify user friendly error message contains number of inputs and their names
        with self.assertRaisesRegex(Error, '.*2.*inputs.*input1.*input2.*'):
            process_function(ov_function=function, argv=argv)

    def test_incompatible_layout(self):
        function = create_function2(shape1=[1, 224, 224, 3], shape2=[1, 4, 224, 224])
        with self.assertRaisesRegex(Exception, '.*input1.*'):
            function.get_parameters()[0].layout = Layout("NDHWC")

    def test_guess_layout_reverse_channels_dont_apply_to_4(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None)
        function = create_function2(shape1=[1, 224, 224, 3], shape2=[1, 4, 224, 224])
        process_function(ov_function=function, argv=argv)

        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() == 'Relu')

    def test_error_guess_layout_reverse_channels_multi_3(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None)
        function = create_function2(shape1=[1, 224, 224, 3], shape2=[1, 3, 3, 224])
        process_function(ov_function=function, argv=argv)
        # Applied to only input1
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() == 'Relu')


    def test_no_guess_layout_reverse_channels_has_layout_no_c(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None)
        function = create_function2(shape1=[1, 224, 224, 3], shape2=[1, 3, 224, 224])
        function.get_parameters()[0].layout = Layout("NHW?")
        function.get_parameters()[1].layout = Layout("N?HW")
        # no suitable inputs
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_guess_layout_reverse_channels_incorrect_pos(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None)
        function = create_function2(shape1=[1, 4, 224, 224], shape2=[1, 224, 224, 2])
        function.get_parameters()[0].layout = Layout("NCHW")
        function.get_parameters()[1].layout = Layout("NHWC")
        # no suitable inputs
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_no_reverse_channels_even_with_layout(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None)
        function = create_function2(shape1=[3, 4, 224, 224], shape2=[1, 224, 3, 224])
        # no suitable inputs
        with self.assertRaises(Exception):
            process_function(ov_function=function, argv=argv)

    def test_reverse_channels_and_mean_scale(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values={
                                                        'input2a': {
                                                           'mean': np.array([1., 2., 3.]),
                                                           'scale': np.array([2., 4., 8.])}},
                         scale=None)
        function = create_function2(shape2=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)

        # Verify that first is gather, then subtract 'mean', then 'scale'
        gather = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(gather.get_type_name() == 'Gather')
        range_node = gather.input(1).get_source_output().get_node()
        self.assertTrue(range_node.get_type_name() == 'Range')
        start = range_node.input(0).get_source_output().get_node()
        end = range_node.input(1).get_source_output().get_node()
        step = range_node.input(2).get_source_output().get_node()
        self.check_constant(start, expected=[2], shape=[])
        self.check_constant(end, expected=[-1], shape=[])
        self.check_constant(step, expected=[-1], shape=[])
        axes = gather.input(2).get_source_output().get_node()
        self.check_constant(axes, expected=[1], shape=[1])

        op_node = list(gather.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[1., 2., 3.], shape=[1, 3, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        # Verify that input1 is not affected
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertEqual(op_node.get_type_name(), 'Relu')

        # Verify that guessed layout (?C??) is not appeared in input2
        self.assertEqual(function.get_parameters()[1].layout, Layout())

    def test_friendly_name(self):
        argv = Namespace(mean_scale_values={'input1': {'mean': np.array([2., 4., 8.]), 'scale': None}},
                         layout_values={'input1': {'source_layout': 'nchw'}},
                         scale=None)
        function = create_function1(shape1=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        # Verify that layout (nchw) is appeared in input1
        self.assertEqual(function.get_parameters()[0].layout, Layout('nchw'))

    def test_sorting_tensor_names(self):
        argv = Namespace(mean_scale_values={'c_input': {'mean': np.array([2., 4., 8.]), 'scale': None}},
                         layout_values={'c_input': {'source_layout': 'nchw'}},
                         scale=127.5)
        function = create_function3(shape1=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=127.5, shape=[1])

        # Verify that layout (nchw) is appeared in input1
        self.assertEqual(function.get_parameters()[0].layout, Layout('nchw'))

    def test_sorting_tensor_names_friendly_name_case(self):
        argv = Namespace(mean_scale_values={'input1': {'mean': np.array([2., 4., 8.]), 'scale': None}},
                         layout_values={'input1': {'source_layout': 'nchw'}},
                         scale=127.5)
        function = create_function3(shape1=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=127.5, shape=[1])

        # Verify that layout (nchw) is appeared in input1
        self.assertEqual(function.get_parameters()[0].layout, Layout('nchw'))

    def test_sorting_tensor_names_unnamed_layout(self):
        argv = Namespace(mean_scale_values={'input1': {'mean': np.array([2., 4., 8.]), 'scale': None}},
                         layout_values={'': {'source_layout': 'nchw'}},
                         scale=127.5)
        function = create_function3(shape1=[1, 3, 224, 224])
        process_function(ov_function=function, argv=argv)
        op_node = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Subtract' or op_node.get_type_name() == 'Add')
        self.check_mean_constant(op_node, expected=[2., 4., 8.], shape=[1, 3, 1, 1])

        op_node = list(op_node.output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node.get_type_name() == 'Divide' or op_node.get_type_name() == 'Multiply')
        self.check_scale_constant(op_node, expected=127.5, shape=[1])

        # Verify that layout (nchw) is appeared in input1
        self.assertEqual(function.get_parameters()[0].layout, Layout('nchw'))

    def test_sorting_tensor_names_unnamed_layout_list(self):
        argv = Namespace(reverse_input_channels=True, mean_scale_values=None, scale=None,
                         layout_values=[{'source_layout': 'nchw', 'target_layout': 'nhwc'},
                                        {'source_layout': 'nhwc', 'target_layout': 'nchw'}])

        function = create_function2(shape1=[1, 3, 5, 5], shape2=[1, 5, 5, 3])
        process_function(ov_function=function, argv=argv)
        # Verify that reverse_channels are applied.
        op_node0 = list(function.get_parameters()[0].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node0.get_type_name() != 'Relu')
        op_node1 = list(function.get_parameters()[1].output(0).get_target_inputs())[0].get_node()
        self.assertTrue(op_node1.get_type_name() != 'Relu')
