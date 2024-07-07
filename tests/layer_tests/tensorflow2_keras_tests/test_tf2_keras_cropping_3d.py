# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasCropping3D(CommonTF2LayerTest):
    def create_keras_cropping_3d_net(self, cropping, input_names, input_shapes, input_type,
                                     ir_version):
        # create TensorFlow 2 model with Keras Cropping3D operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], dtype=input_type,
                           name=input_names[0])  # Variable-length sequence of ints

        y = tf.keras.layers.Cropping3D(cropping=cropping, input_shape=input_shapes[0][1:])(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(cropping=2, input_names=["x"], input_shapes=[[3, 5, 7, 5, 6]],
             input_type=tf.float32),
        dict(cropping=(1, 2, 1), input_names=["x"], input_shapes=[[2, 3, 7, 5, 6]],
             input_type=tf.float32),
        dict(cropping=((2, 1), (3, 2), (1, 1)), input_names=["x"], input_shapes=[[5, 7, 9, 7, 6]],
             input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_cropping_3d_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                       use_legacy_frontend):
        self._test(*self.create_keras_cropping_3d_net(**params, ir_version=ir_version), ie_device,
                   precision,
                   temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
