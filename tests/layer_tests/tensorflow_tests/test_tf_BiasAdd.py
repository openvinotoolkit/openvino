# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestBiasAdd(CommonTFLayerTest):
    def create_bias_add_placeholder_const_net(self, shape, ir_version):
        """
            Tensorflow net                      IR net

            Placeholder->BiasAdd       =>       Placeholder->Add
                         /                                   /
            Const-------/                       Const-------/

        """
        import tensorflow as tf
        import numpy as np

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            y_shape = shape[-1:]

            x = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            constant_value = np.random.randint(0, 1, y_shape).astype(np.float32)
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            tf.nn.bias_add(x, y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        ref_net = None

        return tf_net, ref_net

    def create_bias_add_2_consts_net(self, shape, ir_version):
        """
            Tensorflow net                         IR net

            Const->BiasAdd-->Concat       =>       Const---->Concat
                    /        /                                  /
            Const--/        /                      Placeholder-/
                           /
            Placeholder---/

        """
        import tensorflow as tf
        import numpy as np

        tf.compat.v1.reset_default_graph()

        tf_concat_axis = -1

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()
            tf_y_shape = tf_x_shape[-1:]

            constant_value_x = np.random.randint(-256, 256, tf_x_shape).astype(np.float32)
            x = tf.constant(constant_value_x)
            constant_value_y = np.random.randint(-256, 256, tf_y_shape).astype(np.float32)
            y = tf.constant(constant_value_y)

            add = tf.nn.bias_add(x, y, name="Operation")

            placeholder = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')  # Input_1 in graph_def

            tf.concat([placeholder, add], axis=tf_concat_axis, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_2D = [
        dict(shape=[2, 1]),
        dict(shape=[2, 224])
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_bias_add_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_2D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_3D = [
        dict(shape=[2, 3, 5]),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_bias_add_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_4D = [
        dict(shape=[2, 3, 5, 7])
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bias_add_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_5D = [
        dict(shape=[2, 3, 5, 7, 9])
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bias_add_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
