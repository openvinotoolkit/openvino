# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasUpSampling3D(CommonTF2LayerTest):
    def create_keras_upsampling3d_net(self, input_names, input_shapes, input_type, size,
                                      data_format, ir_version):
        # create TensorFlow 2 model with Keras UpSampling3D operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0], dtype=input_type)
        y = tf.keras.layers.UpSampling3D(size=size, data_format=data_format)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    # Tests for nearest interpolation
    test_data = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 6, 2]], input_type=tf.float32,
             size=(2, 4, 3), data_format='channels_last'),
        dict(input_names=["x1"], input_shapes=[[3, 2, 4, 2, 2]], input_type=tf.float32,
             size=(2, 3, 1), data_format='channels_last'),
        dict(input_names=["x1"], input_shapes=[[1, 3, 8, 9, 6]], input_type=tf.float32,
             size=(3, 5, 2), data_format='channels_last'),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_upsampling3(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                               use_new_frontend):
        self._test(*self.create_keras_upsampling3d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data_channels_first = [
        pytest.param(
            dict(input_names=["x1"], input_shapes=[[5, 4, 5, 1, 3]], input_type=tf.float32,
                 size=(3, 2, 4), data_format='channels_first'),
            marks=pytest.mark.xfail(reason="49540")),
        pytest.param(
            dict(input_names=["x1"], input_shapes=[[3, 2, 7, 2, 8]], input_type=tf.float32,
                 size=(2, 3, 3), data_format='channels_first'),
            marks=pytest.mark.xfail(reason="49540"))
    ]

    @pytest.mark.parametrize("params", test_data_channels_first)
    @pytest.mark.nightly
    def test_keras_upsampling2d_channels_first(self, params, ie_device, precision, ir_version,
                                               temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_keras_upsampling3d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
