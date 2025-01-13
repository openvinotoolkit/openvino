# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestWhile(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(1, 10, x_shape).astype(np.int32)
        inputs_data['y:0'] = np.random.randint(-50, 50, y_shape).astype(np.int32)
        return inputs_data

    def create_while_net(self, y_shape, data_type, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def while_function(x, y):
            @tf.function
            def cond(x, y):
                return tf.less(x, 10)

            @tf.function
            def body(x, y):
                y_new = tf.add(y, tf.constant(2, dtype=data_type))
                x_new = tf.add(x, 1)
                return x_new, y_new

            return tf.while_loop(cond, body, [x, y])

        tf_while_graph = tf.function(while_function)
        x = np.random.randint(1, 10, []).astype(data_type)
        y = np.random.randint(-50, 50, y_shape).astype(data_type)
        concrete_func = tf_while_graph.get_concrete_function(x, y)

        # lower_control_flow defines representation of While operation
        # in case of lower_control_flow=True it is decomposed into LoopCond, NextIteration and TensorArray operations
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        return graph_def, None

    test_data_basic = [
        dict(y_shape=[2, 3], data_type=np.int32, lower_control_flow=False),
        dict(y_shape=[2, 3], data_type=np.int32, lower_control_flow=True),
        dict(y_shape=[2, 1, 4], data_type=np.int32, lower_control_flow=False),
        dict(y_shape=[2, 1, 4], data_type=np.int32, lower_control_flow=True)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_while_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_while_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestWhileShapeVariant(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(1, 10, x_shape).astype(np.int32)
        inputs_data['y:0'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_while_net(self, y_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def while_function(x, y):
            @tf.function
            def cond(x, y):
                return tf.less(x, 10)

            @tf.function
            def body(x, y):
                add_2 = tf.add(y, tf.constant(2, dtype=tf.float32))
                y_new = tf.concat(values=[y, add_2], axis=0)
                x_new = tf.add(x, tf.constant(1, tf.int32))
                return x_new, y_new

            return tf.while_loop(cond, body, [x, y],
                                 shape_invariants=[tf.TensorShape([]),
                                                   tf.TensorShape([None] + y_shape[1:])])

        tf_while_graph = tf.function(while_function)
        x = np.random.randint(1, 10, []).astype(np.int32)
        y = np.random.randint(-50, 50, y_shape).astype(np.float32)
        concrete_func = tf_while_graph.get_concrete_function(x, y)

        # lower_control_flow defines representation of While operation
        # in case of lower_control_flow=True it is decomposed into LoopCond, NextIteration and TensorArray operations
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        return graph_def, None

    test_data_basic = [
        dict(y_shape=[2, 3], lower_control_flow=False),
        dict(y_shape=[2, 3], lower_control_flow=True),
        dict(y_shape=[2, 1, 4], lower_control_flow=False),
        dict(y_shape=[2, 1, 4], lower_control_flow=True)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_while_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_while_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestWhileWithNestedIf(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(1, 10, x_shape).astype(np.int32)
        inputs_data['y:0'] = np.random.randint(-50, 50, y_shape).astype(np.int32)
        return inputs_data

    def create_while_with_nested_if_net(self, y_shape, data_type, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def while_function(x, y):
            @tf.function
            def cond(x, y):
                return tf.less(x, 10)

            @tf.function
            def body(x, y):
                # create If operation inside While body
                # use different logic for updating y based on x
                def if_op(cond, y):
                    def then_branch():
                        y_new = tf.multiply(y, tf.constant(2, dtype=data_type))
                        return y_new

                    def else_branch():
                        y_new = tf.subtract(y, tf.constant(55, dtype=data_type))
                        return y_new

                    if_op = tf.cond(cond, then_branch, else_branch)
                    output = tf.identity(if_op, name='if_op')
                    return output

                y_new = tf.add(y, tf.constant(2, dtype=data_type))
                cond = tf.less(x, 5)
                y_new = if_op(cond, y_new)
                x_new = tf.add(x, 1)
                return x_new, y_new

            return tf.while_loop(cond, body, [x, y])

        tf_while_graph = tf.function(while_function)
        x = np.random.randint(9, 10, []).astype(data_type)
        y = np.random.randint(-50, 50, y_shape).astype(data_type)
        concrete_func = tf_while_graph.get_concrete_function(x, y)

        # lower_control_flow defines representation of While operation
        # in case of lower_control_flow=True it is decomposed into LoopCond, NextIteration and TensorArray operations
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        return graph_def, None

    test_data_basic = [
        dict(y_shape=[2, 3], data_type=np.int32, lower_control_flow=False),
        dict(y_shape=[2, 3], data_type=np.int32, lower_control_flow=True),
        dict(y_shape=[2, 1, 4], data_type=np.int32, lower_control_flow=False),
        dict(y_shape=[2, 1, 4], data_type=np.int32, lower_control_flow=True)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_while_with_nested_if_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                        use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("accuracy issue on GPU")
        self._test(*self.create_while_with_nested_if_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
