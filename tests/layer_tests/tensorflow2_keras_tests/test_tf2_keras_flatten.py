# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasFlatten(CommonTF2LayerTest):
    def create_keras_flatten_net(self, input_names, input_shapes, input_type, data_format,
                                 ir_version):
        """
                create TensorFlow 2 model with Keras Flatten operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.Flatten(data_format)(x)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None
        return tf2_net, ref_net

    test_data_float32 = [
        dict(input_names=["x"], input_shapes=[[5, 4, 3]], input_type=tf.float32, data_format=None),
        dict(input_names=["x"], input_shapes=[[5, 4, 8, 16]], input_type=tf.float32,
             data_format="channels_first"),
        dict(input_names=["x"], input_shapes=[[5, 1, 16, 8, 3]], input_type=tf.float32,
             data_format="channels_last"),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_flatten_float32(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_keras_flatten_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
