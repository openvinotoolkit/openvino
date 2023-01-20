# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tflite_layer_test_class import TFLiteLayerTest


class TestAdd(TFLiteLayerTest):

    @staticmethod
    def make_model(x_shape, y_shape):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            tf_x_shape = x_shape.copy()
            tf_y_shape = y_shape.copy()

            x = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            constant_value = np.random.randint(-256, 256, tf_y_shape).astype(np.float32)
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            add = tf.add(x, y, name="Operation")
            add_shape = add.shape.as_list()

            tf.compat.v1.global_variables_initializer()
            net = sess.graph_def
        return net

    test_data_1D = [
        dict(x_shape=[1], y_shape=[1]),
        pytest.param(dict(x_shape=[3], y_shape=[3]), marks=pytest.mark.xfail(reason="*-19180"))
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_add(self, params, ie_device, precision, temp_dir):
        self._test(ie_device, precision, temp_dir, **params)
