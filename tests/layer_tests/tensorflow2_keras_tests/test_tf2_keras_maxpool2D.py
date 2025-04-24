# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasMaxPool2D(CommonTF2LayerTest):
    def create_keras_maxpool2D_net(self, input_names, input_shapes, input_type, pool_size, strides,
                                   padding, dataformat,
                                   ir_version):
        """
                create TensorFlow 2 model with Keras MaxPool2D operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding,
                                      data_format=dataformat)(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None
        return tf2_net, ref_net

    test_data_float32 = [
        dict(input_names=["x"], input_shapes=[[5, 4, 2, 6]], input_type=tf.float32,
             pool_size=(1, 1),
             strides=None, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[5, 8, 4, 8]], input_type=tf.float32,
             pool_size=(3, 4),
             strides=None, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[5, 3, 6, 4]], input_type=tf.float32,
             pool_size=(2, 2),
             strides=None, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[5, 4, 5, 12]], input_type=tf.float32, pool_size=1,
             strides=4, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[5, 4, 6, 6]], input_type=tf.float32,
                          pool_size=(2, 3),
                          strides=(3, 3), padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[5, 4, 4, 8]], input_type=tf.float32, pool_size=2,
             strides=2, padding='valid', dataformat='channels_last'),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_maxpool2D_pool_strides_float32(self, params, ie_device, precision, temp_dir,
                                                  ir_version, use_legacy_frontend):
        self._test(*self.create_keras_maxpool2D_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
