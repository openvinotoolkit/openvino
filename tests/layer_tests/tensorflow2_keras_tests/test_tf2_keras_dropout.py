# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasDropout(CommonTF2LayerTest):
    def create_keras_dropout_net(self, input_names, input_shapes, input_type, rate, noise_shape,
                                 ir_version):
        """
               create TensorFlow 2 model with Keras Dropout operation
        """

        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.Dropout(rate=rate, noise_shape=noise_shape, seed=4)(x, training=False)

        tf2_net = tf.keras.Model(inputs=[x], outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more important and needs to be checked
        #  in the first
        ref_net = None
        return tf2_net, ref_net

    test_data_float32_precommit = [
        dict(input_names=["x"], input_shapes=[[5, 4, 4, 12, 6]], input_type=tf.float32, rate=.5,
             noise_shape=(4, 4, 12, 6)),
    ]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    def test_keras_dropout_float32(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                                   use_new_frontend):
        self._test(*self.create_keras_dropout_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data_float32 = [
        dict(input_names=["x"], input_shapes=[[5, 2]], input_type=tf.float32, rate=0.1,
             noise_shape=(2,)),
        dict(input_names=["x"], input_shapes=[[5, 4, 4]], input_type=tf.float32, rate=0.9,
             noise_shape=(4,)),
        dict(input_names=["x"], input_shapes=[[5, 4, 4]], input_type=tf.float32, rate=0.4,
             noise_shape=(4, 4)),
        dict(input_names=["x"], input_shapes=[[5, 4, 4]], input_type=tf.float32, rate=0.,
             noise_shape=(4, 4)),
        dict(input_names=["x"], input_shapes=[[5, 4, 4, 4]], input_type=tf.float32, rate=.999,
             noise_shape=(4, 4, 4)),
        dict(input_names=["x"], input_shapes=[[5, 4, 4, 12, 6]], input_type=tf.float32, rate=.5,
             noise_shape=(4, 4, 12, 6)),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_dropout_float32(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                                   use_new_frontend):
        self._test(*self.create_keras_dropout_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
