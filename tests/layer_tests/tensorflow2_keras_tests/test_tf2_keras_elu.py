# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasELU(CommonTF2LayerTest):
    def create_keras_elu_net(self, input_names, input_shapes, input_type, alpha, ir_version):
        """
               Tensorflow2 Keras net:                     IR net:
                      Input               =>               Input
                        |                                    |
                       ELU                                  Elu
        """
        # create TensorFlow 2 model with Keras ELU operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:],
                            name=input_names[0])  # Variable-length sequence of ints
        y = tf.keras.layers.ELU(alpha=alpha)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # create reference IR net
        ref_net = None

        return tf2_net, ref_net

    test_data_float32_precommit = [dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]],
                                        input_type=tf.float32, alpha=1.0)]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_elu_float32(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_keras_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_float32 = [
        dict(input_names=["x1"], input_shapes=[[5, 4]], input_type=tf.float32, alpha=1.0),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8]], input_type=tf.float32, alpha=1.0),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]], input_type=tf.float32, alpha=1.0),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]], input_type=tf.float32, alpha=1.0)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_elu_float32(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_keras_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_float32_alpha2 = [dict(input_names=["x1"], input_shapes=[[5, 4]],
                                     input_type=tf.float32, alpha=2.0),
                                dict(input_names=["x1"], input_shapes=[[5, 4, 8]],
                                     input_type=tf.float32, alpha=3.0),
                                dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
                                     input_type=tf.float32, alpha=4.0),
                                dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]],
                                     input_type=tf.float32, alpha=5.0)]

    @pytest.mark.parametrize("params", test_data_float32_alpha2)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_elu_float32_alpha2(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_keras_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
