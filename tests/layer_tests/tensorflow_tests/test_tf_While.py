# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestWhile(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['x'] = np.random.randint(1, 10, x_shape).astype(np.int32)
        inputs_data['y'] = np.random.randint(-50, 50, y_shape).astype(np.int32)
        return inputs_data

    def create_while_net(self, y_shape, data_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(data_type, [], 'x')
            y = tf.compat.v1.placeholder(data_type, y_shape, 'y')

            @tf.function
            def cond(x, y):
                return tf.less(x, 10)

            @tf.function
            def body(x, y):
                y_new = tf.add(y, tf.constant(2, dtype=data_type))
                x_new = tf.add(x, 1)
                return x_new, y_new

            tf.while_loop(cond, body, [x, y])
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(y_shape=[2, 3], data_type=tf.int32),
        dict(y_shape=[2, 1, 4], data_type=tf.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    def test_while_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_new_frontend, use_old_api):
        self._test(*self.create_while_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestWhileShapeVariant(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x']
        y_shape = inputs_info['y']
        inputs_data = {}
        inputs_data['x'] = np.random.randint(1, 10, x_shape).astype(np.int32)
        inputs_data['y'] = np.random.randint(-50, 50, y_shape).astype(np.float32)
        return inputs_data

    def create_while_net(self, y_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.int32, [], 'x')
            y = tf.compat.v1.placeholder(tf.float32, y_shape, 'y')

            @tf.function
            def cond(x, y):
                return tf.less(x, 10)

            @tf.function
            def body(x, y):
                add_2 = tf.add(y, tf.constant(2, dtype=tf.float32))
                y_new = tf.concat(values=[y, add_2], axis=0)
                x_new = tf.add(x, tf.constant(1, tf.int32))
                return x_new, y_new

            tf.while_loop(cond, body, [x, y],
                          shape_invariants=[tf.TensorShape([]),
                                            tf.TensorShape([None] + y_shape[1:])])
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(y_shape=[2, 3]),
        dict(y_shape=[2, 1, 4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    def test_while_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_new_frontend, use_old_api):
        self._test(*self.create_while_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
