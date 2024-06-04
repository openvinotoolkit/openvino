# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasZeroPadding2D(CommonTF2LayerTest):
    def create_keras_zeropadding2d_net(self, input_names, input_shapes, input_type, padding,
                                       data_format, ir_version):
        # create TensorFlow 2 model with Keras ZeroPadding2D operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0], dtype=input_type)
        y = tf.keras.layers.ZeroPadding2D(padding=padding, data_format=data_format)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_channels_last = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]], input_type=tf.float32,
             padding=2, data_format='channels_last'),
        dict(input_names=["x1"], input_shapes=[[3, 2, 4, 6]], input_type=tf.float32,
             padding=(3, 0), data_format='channels_last'),
        dict(input_names=["x1"], input_shapes=[[1, 3, 8, 7]], input_type=tf.float32,
             padding=((5, 1), (3, 4)), data_format='channels_last'),
    ]

    @pytest.mark.parametrize("params", test_data_channels_last)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_zeropadding2d_channels_last(self, params, ie_device, precision, ir_version,
                                               temp_dir):
        self._test(*self.create_keras_zeropadding2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)

    test_data_channels_first = [
        dict(input_names=["x1"], input_shapes=[[3, 8, 4, 5]], input_type=tf.float32,
             padding=1, data_format='channels_first'),
        dict(input_names=["x1"], input_shapes=[[6, 4, 2, 3]], input_type=tf.float32,
             padding=(3, 0), data_format='channels_first'),
        dict(input_names=["x1"], input_shapes=[[7, 8, 3, 4]], input_type=tf.float32,
             padding=((0, 0), (3, 4)), data_format='channels_first'),
    ]

    @pytest.mark.parametrize("params", test_data_channels_first)
    @pytest.mark.nightly
    def test_keras_zeropadding2d_channels_first(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_zeropadding2d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
