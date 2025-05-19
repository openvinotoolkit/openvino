# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasTimeDistributed(CommonTF2LayerTest):
    def create_keras_timedistributed_net(self, input_names, input_shapes, input_type, conv_2d_layer,
                                         ir_version):
        conv_layers_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.keras.layers.<cell> operation have no "==" operation to be compared
            "Conv2D": tf.keras.layers.Conv2D(64, (3, 3))
        }
        conv_2d_layer = conv_layers_structure[conv_2d_layer]

        # create TensorFlow 2 model with Keras TimeDistributed operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.TimeDistributed(conv_2d_layer)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data = [
        dict(input_names=["x1"], input_shapes=[[2, 10, 128, 128, 3]], input_type=tf.float32,
             conv_2d_layer="Conv2D"),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_timedistributed(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_keras_timedistributed_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
