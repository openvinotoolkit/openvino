# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasSubtract(CommonTF2LayerTest):
    def create_keras_subtract_net(self, input_names, input_shapes, input_type, ir_version):
        """
               Tensorflow2 Keras net:                     IR net:
                 Input1    Input2            =>      Input1     Input2
                   \         /                           \       /
                    Subtract                              Subtract
        """
        # create TensorFlow 2 model with Keras Subtract operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        inputs = []
        for ind in range(len(input_names)):
            inputs.append(tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind]))
        y = tf.keras.layers.Subtract()(inputs)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # create reference IR net
        ref_net = None

        return tf2_net, ref_net

    test_data_float32_precommit = [
        dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
             input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    def test_keras_subtract_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        self._test(*self.create_keras_subtract_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_float32 = [dict(input_names=["x1", "x2"], input_shapes=[[5, 4], [5, 4]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8], [5, 4, 8]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8, 2], [5, 4, 8, 2]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2"],
                              input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
                              input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_subtract_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        self._test(*self.create_keras_subtract_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
