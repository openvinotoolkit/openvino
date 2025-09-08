# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasMaxPool3D(CommonTF2LayerTest):
    def create_keras_maxpool3D_net(self, input_names, input_shapes, input_type, pool_size, strides,
                                   padding, dataformat, ir_version):
        """
                create TensorFlow 2 model with Keras MaxPool3D operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.MaxPool3D(pool_size=pool_size, strides=strides, padding=padding,
                                      data_format=dataformat)(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None
        return tf2_net, ref_net

    test_data_float32 = [
        dict(input_names=["x"], input_shapes=[[2, 2, 9, 4, 3]], input_type=tf.float32, pool_size=1,
             strides=None, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[2, 3, 4, 5, 6]], input_type=tf.float32, pool_size=2,
             strides=None, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[2, 3, 4, 5, 6]], input_type=tf.float32,
             pool_size=(2, 3, 4),
             strides=None, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[2, 3, 5, 7, 9]], input_type=tf.float32, pool_size=1,
             strides=(2, 2, 2), padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[5, 4, 4, 4, 4]], input_type=tf.float32, pool_size=2,
             strides=(2, 3, 2), padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[2, 6, 8, 4, 12]], input_type=tf.float32,
             pool_size=(3, 2, 4),
             strides=(3, 4, 2), padding='valid', dataformat='channels_last'),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_maxpool3D_pool_strides_float32(self, params, ie_device, precision, temp_dir,
                                                  ir_version):
        self._test(*self.create_keras_maxpool3D_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)

    test_data_p_dformat_float32 = [
        dict(input_names=["x"], input_shapes=[[2, 2, 4, 6, 8]], input_type=tf.float32, pool_size=1,
             strides=1, padding='valid', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[2, 2, 4, 6, 8]], input_type=tf.float32, pool_size=2,
             strides=2, padding='same', dataformat='channels_last'),
        dict(input_names=["x"], input_shapes=[[2, 2, 4, 6, 8]], input_type=tf.float32, pool_size=4,
             strides=(2, 3, 2), padding='valid', dataformat='channels_first'),
        dict(input_names=["x"], input_shapes=[[2, 2, 4, 6, 8]], input_type=tf.float32, pool_size=2,
             strides=3, padding='same', dataformat='channels_first'),
    ]

    @pytest.mark.parametrize("params", test_data_p_dformat_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_maxpool3D_padding_and_data_format(self, params, ie_device, precision, temp_dir,
                                                     ir_version):
        self._test(*self.create_keras_maxpool3D_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
