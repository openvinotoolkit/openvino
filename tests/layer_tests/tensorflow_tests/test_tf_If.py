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

    def create_if_net(self, x_shape, y_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            cond = tf.compat.v1.placeholder(tf.bool, [], 'cond')
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            y = tf.compat.v1.placeholder(tf.float32, y_shape, 'y')

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
            tf.identity(if_output[0], name='output1')
            tf.identity(if_output[1], name='output2')
            tf.identity(if_output[2], name='output3')
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[3], y_shape=[2, 3]),
        dict(x_shape=[2, 1, 4], y_shape=[2, 1, 4]),
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

    def create_if_net(self, ind_shape, y_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            cond = tf.compat.v1.placeholder(tf.bool, [], 'cond')
            ind = tf.compat.v1.placeholder(tf.int32, ind_shape, 'ind')
            y = tf.compat.v1.placeholder(tf.float32, y_shape, 'y')

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
            tf.identity(if_output[0], name='output1')
            tf.identity(if_output[1], name='output2')
            tf.identity(if_output[2], name='output3')
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(ind_shape=[3], y_shape=[2, 3]),
        dict(ind_shape=[2, 1, 4], y_shape=[2, 1, 4]),
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
