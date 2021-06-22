# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class Testmul(CommonTFLayerTest):
    disable_input_layout_conversion = True

    def create_mul_placeholder_const_net(self, x_shape, y_shape, ir_version):
        """
            Tensorflow net                  IR net

            Placeholder->mul       =>       Placeholder->mul
                         /                               /
            Const-------/                   Const-------/

        """
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'Input')
            constant_value = np.random.randint(-256, 256, y_shape).astype(np.float32)
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            tf.multiply(x, y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # TODO: implement tests for 2 Consts + mul

    test_data_1D = [
        dict(x_shape=[1], y_shape=[1]),
        dict(x_shape=[3], y_shape=[3]),
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_2D = [
        dict(x_shape=[1, 1], y_shape=[1, 1]),
        dict(x_shape=[1, 3], y_shape=[1, 3]),
        dict(x_shape=[3, 1], y_shape=[3, 1]),
        dict(x_shape=[2, 3], y_shape=[2, 3])
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_3D = [
        dict(x_shape=[1, 1, 1], y_shape=[1, 1, 1]),
        dict(x_shape=[1, 3, 1], y_shape=[1, 3, 1]),
        dict(x_shape=[1, 1, 3], y_shape=[1, 1, 3]),
        dict(x_shape=[1, 3, 5], y_shape=[1, 3, 5]),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1, 3, 1, 1]),
        dict(x_shape=[1, 1, 1, 3], y_shape=[1, 1, 1, 3]),
        dict(x_shape=[1, 3, 5, 7], y_shape=[1, 3, 5, 7]),
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mul_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1, 1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[1, 3, 1, 1, 1]),
        dict(x_shape=[1, 1, 1, 1, 3], y_shape=[1, 1, 1, 1, 3]),
        dict(x_shape=[1, 3, 5, 7, 9], y_shape=[1, 3, 5, 7, 9]),
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mul_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir)

    ###############################################################################################
    #                                                                                             #
    #                                       Broadcast cases                                       #
    #                                                                                             #
    ###############################################################################################

    test_data_broadcast_1D = [
        dict(x_shape=[3], y_shape=[1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_1D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_1D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir)

    test_data_broadcast_2D = [
        dict(x_shape=[1, 1], y_shape=[1]),
        dict(x_shape=[1, 3], y_shape=[1]),
        dict(x_shape=[1, 3], y_shape=[3]),
        dict(x_shape=[3, 1], y_shape=[3]),
        dict(x_shape=[3, 1], y_shape=[1, 3, 1, 1]),
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_2D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_2D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir)

    test_data_broadcast_3D = [
        dict(x_shape=[1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1], y_shape=[3]),
        dict(x_shape=[1, 3, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[3, 1, 5], y_shape=[1, 3, 5]),
        dict(x_shape=[2, 3, 1], y_shape=[1, 3, 2]),
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_3D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir)

    test_data_broadcast_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 3, 5, 7], y_shape=[7]),
        dict(x_shape=[1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 3, 1, 2], y_shape=[3, 1, 2]),
        dict(x_shape=[1, 3, 1, 2], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 3, 5, 7], y_shape=[1, 1, 1, 7]),
        dict(x_shape=[2, 3, 1, 2], y_shape=[2, 3, 1, 2])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mul_placeholder_const_broadcast_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir)

    test_data_broadcast_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[7, 5]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[5]),
        dict(x_shape=[1, 1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[7, 1]),
        dict(x_shape=[1, 3, 1, 1, 2], y_shape=[1, 5, 2]),
        dict(x_shape=[1, 3, 5, 1, 2], y_shape=[3, 5, 1, 2]),
        dict(x_shape=[1, 3, 5, 7, 9], y_shape=[1, 1, 5, 1, 9]),
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_5D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version), ie_device, precision,
                   ir_version=ir_version, temp_dir=temp_dir)
