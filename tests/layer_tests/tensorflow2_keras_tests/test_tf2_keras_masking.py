# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasMasking(CommonTF2LayerTest):
    def create_keras_masking_net(self, input_names, input_shapes, input_type, mask_value,
                                 ir_version):
        """
                create TensorFlow 2 model with Keras Masking operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        l1 = tf.keras.layers.Dense(4)(x)
        y = tf.keras.layers.Masking(mask_value=mask_value)(l1)
        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None
        return tf2_net, ref_net

    test_data_float32 = [
        dict(input_names=["x"], input_shapes=[[5, 4, 8]], input_type=tf.float32, mask_value=0.),
        dict(input_names=["x"], input_shapes=[[5, 4, 8, 16]], input_type=tf.float32, mask_value=1.),
        dict(input_names=["x"], input_shapes=[[9, 4, 8, 47]], input_type=tf.float32,
             mask_value=0.3),
        dict(input_names=["x"], input_shapes=[[8, 4, 32]], input_type=tf.float32, mask_value=-132.),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="49567")
    def test_keras_masking_float32(self, params, ie_device, precision, temp_dir, ir_version):
        self._test(*self.create_keras_masking_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
