# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestIfFloat(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `y`"
        cond_shape = inputs_info['cond']
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['cond'] = np.random.randint(0, 2, cond_shape).astype(bool)
        inputs_data['x'] = np.random.randint(1, 10, x_shape).astype(np.float32)
        inputs_data['y'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_if_net(self, x_shape, y_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def if_function(cond, x, y):
            def then_branch():
                output1 = tf.add(x, y)
                output2 = tf.multiply(x, y)
                output3 = tf.subtract(x, y)
                return output1, output2, output3

            def else_branch():
                const_two = tf.constant(2.0, dtype=tf.float32)
                output1 = tf.add(y, const_two)
                output2 = tf.multiply(const_two, y)
                output3 = x - y + const_two
                return output1, output2, output3

            if_output = tf.cond(cond, then_branch, else_branch)
            output1 = tf.identity(if_output[0], name='output1')
            output2 = tf.identity(if_output[1], name='output2')
            output3 = tf.identity(if_output[2], name='output3')
            return output1, output2, output3

        tf_if_graph = tf.function(if_function)
        cond = np.random.randint(0, 2, []).astype(bool)
        x = np.random.randint(1, 10, x_shape).astype(np.float32)
        y = np.random.randint(-50, 50, y_shape).astype(np.float32)
        concrete_func = tf_if_graph.get_concrete_function(cond, x, y)

        # lower_control_flow defines representation of If operation
        # in case of lower_control_flow=True it is decomposed into Switch and Merge nodes
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        tf_net = frozen_func.graph.as_graph_def(add_shapes=True)
        return tf_net, None

    test_data_basic = [
        dict(x_shape=[3], y_shape=[2, 3], lower_control_flow=False),
        dict(x_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=False),
        pytest.param(dict(x_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=True),
                     marks=pytest.mark.xfail(reason="92632"))
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_if_basic(self, params, ie_device, precision, ir_version, temp_dir,
                      use_new_frontend, use_old_api):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        self._test(*self.create_if_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestIfInt(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'cond' in inputs_info, "Test error: inputs_info must contain `cond`"
        assert 'ind' in inputs_info, "Test error: inputs_info must contain `ind`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `y`"
        cond_shape = inputs_info['cond']
        ind_shape = inputs_info['ind']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['cond'] = np.random.randint(0, 2, cond_shape).astype(bool)
        inputs_data['ind'] = np.random.randint(1, 10, ind_shape).astype(np.int32)
        inputs_data['y'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_if_net(self, ind_shape, y_shape, lower_control_flow):
        from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
        def if_function(cond, ind, y):
            def then_branch():
                const_one = tf.constant(1, dtype=tf.int32)
                output1 = tf.add(ind, const_one)
                output2 = tf.multiply(tf.cast(output1, tf.float32), y)
                output3 = tf.subtract(tf.cast(output1, tf.float32), y)
                return output1, output2, output3

            def else_branch():
                const_two = tf.constant(2, dtype=tf.int32)
                output1 = tf.add(ind, const_two)
                output2 = tf.multiply(tf.cast(output1, tf.float32), y)
                output3 = tf.cast(output1, tf.float32) - y
                return output1, output2, output3

            if_output = tf.cond(cond, then_branch, else_branch)
            output1 = tf.identity(if_output[0], name='output1')
            output2 = tf.identity(if_output[1], name='output2')
            output3 = tf.identity(if_output[2], name='output3')
            return output1, output2, output3

        tf_if_graph = tf.function(if_function)
        cond = np.random.randint(0, 2, []).astype(bool)
        ind = np.random.randint(1, 10, ind_shape).astype(np.int32)
        y = np.random.randint(-50, 50, y_shape).astype(np.float32)
        concrete_func = tf_if_graph.get_concrete_function(cond, ind, y)

        # lower_control_flow defines representation of If operation
        # in case of lower_control_flow=True it is decomposed into Switch and Merge nodes
        frozen_func = convert_variables_to_constants_v2(concrete_func,
                                                        lower_control_flow=lower_control_flow)

        graph_def = frozen_func.graph.as_graph_def(add_shapes=True)
        return graph_def, None

    test_data_basic = [
        dict(ind_shape=[3], y_shape=[2, 3], lower_control_flow=False),
        dict(ind_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=False),
        pytest.param(dict(ind_shape=[2, 1, 4], y_shape=[2, 1, 4], lower_control_flow=True),
                     marks=pytest.mark.xfail(reason="92632"))

    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_if_basic(self, params, ie_device, precision, ir_version, temp_dir,
                      use_new_frontend, use_old_api):
        if ie_device == 'GPU':
            pytest.xfail('104855')
        self._test(*self.create_if_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
