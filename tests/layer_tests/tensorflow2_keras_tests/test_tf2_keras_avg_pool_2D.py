# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasAvgPool2D(CommonTF2LayerTest):
    def create_keras_avg_pool_2D_net(self, pool_size, strides, padding, data_format, input_names,
                                     input_shapes,
                                     input_type, ir_version):
        # create TensorFlow 2 model with Keras AvgPool2D operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state

        inputs = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])

        y = tf.keras.layers.AvgPool2D(
            pool_size=pool_size, strides=strides, padding=padding, data_format=data_format)(inputs)
        tf2_net = tf.keras.Model(inputs=[inputs], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(pool_size=(3, 3), strides=(2, 2), padding='valid', data_format='channels_last',
             input_names=["x1"],
             input_shapes=[[3, 4, 5, 6]], input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_avg_pool_2D_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                       use_legacy_frontend):
        self._test(*self.create_keras_avg_pool_2D_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_extended_float32 = [
        dict(pool_size=(3, 3), strides=None, padding='same', data_format='channels_last',
             input_names=["x1"],
             input_shapes=[[3, 4, 5, 6]], input_type=tf.float32),
        dict(pool_size=(5, 5), strides=None, padding='same', data_format='channels_last',
             input_names=["x1"],
             input_shapes=[[3, 4, 5, 6]], input_type=tf.float32),
        dict(pool_size=(5, 5), strides=(3, 3), padding='valid', data_format='channels_last',
             input_names=["x1"],
             input_shapes=[[3, 7, 6, 5]], input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_extended_float32)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_avg_pool_2D_extended_float32(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_avg_pool_2D_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
