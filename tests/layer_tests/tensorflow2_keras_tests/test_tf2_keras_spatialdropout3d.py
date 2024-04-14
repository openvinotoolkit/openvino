# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasSpatialDropout3D(CommonTF2LayerTest):
    def create_keras_spatialdropout3d_net(self, input_names, input_shapes, input_type, rate,
                                          data_format, ir_version):
        # create TensorFlow 2 model with Keras SpatialDropout3D operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0], dtype=input_type)
        y = tf.keras.layers.SpatialDropout3D(rate=rate, data_format=data_format)(x1, False)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 3, 2, 3]], input_type=tf.float32, rate=0.0,
             data_format='channels_last'),
        dict(input_names=["x1"], input_shapes=[[5, 4, 3, 2, 3]], input_type=tf.float32, rate=0.5,
             data_format='channels_last'),
        dict(input_names=["x1"], input_shapes=[[3, 2, 6, 1, 3]], input_type=tf.float32, rate=0.8,
             data_format='channels_last'),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_spatialdropout3d(self, params, ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        pytest.skip("Error: failed due to missing a required argument: x1")
        self._test(*self.create_keras_spatialdropout3d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_channels_first = [
        dict(input_names=["x1"], input_shapes=[[3, 2, 3, 4, 5]], input_type=tf.float32, rate=0.0,
             data_format='channels_first'),
        dict(input_names=["x1"], input_shapes=[[3, 2, 3, 4, 1]], input_type=tf.float32, rate=0.5,
             data_format='channels_first'),
        dict(input_names=["x1"], input_shapes=[[2, 3, 1, 6, 3]], input_type=tf.float32, rate=0.8,
             data_format='channels_first'),
    ]

    @pytest.mark.parametrize("params", test_data_channels_first)
    @pytest.mark.nightly
    def test_keras_spatialdropout3d_channels_first(self, params, ie_device, precision, ir_version,
                                                   temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_spatialdropout3d_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
