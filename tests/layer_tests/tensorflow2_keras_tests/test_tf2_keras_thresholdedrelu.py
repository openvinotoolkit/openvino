# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasThresholdedReLU(CommonTF2LayerTest):
    def create_keras_thresholdedrelu_net(self, input_names, input_shapes, input_type, theta,
                                         ir_version):
        # create TensorFlow 2 model with Keras ThresholdedReLU operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.ThresholdedReLU(theta=theta)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_precommit = [
        dict(input_names=["x1"], input_shapes=[[3, 5, 3, 5, 2]], input_type=tf.float32, theta=0.2),
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_keras_thresholdedrelu_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_old_api, use_new_frontend):
        self._test(*self.create_keras_thresholdedrelu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data = [
        dict(input_names=["x1"], input_shapes=[[5, 2]], input_type=tf.float32, theta=0.0),
        dict(input_names=["x1"], input_shapes=[[1, 4, 3]], input_type=tf.float32, theta=1.0),
        dict(input_names=["x1"], input_shapes=[[2, 5, 3, 6]], input_type=tf.float32, theta=10.0),
        dict(input_names=["x1"], input_shapes=[[3, 5, 3, 5, 2]], input_type=tf.float32, theta=0.2),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_keras_thresholdedrelu_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_old_api, use_new_frontend):
        self._test(*self.create_keras_thresholdedrelu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
