# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasReshape(CommonTF2LayerTest):
    def create_keras_reshape_net(self, input_names, input_shapes, input_type, target_shape,
                                 ir_version):
        # create TensorFlow 2 model with Keras Reshape operation
        tf.keras.backend.clear_session()
        x1 = tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0])
        y = tf.keras.layers.Reshape(target_shape=target_shape)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: create a reference net. Now it is ommitted and tests only inference since it is more important
        ref_net = None

        return tf2_net, ref_net

    test_data = [
        dict(input_names=["x1"], input_shapes=[[5, 4]], input_type=tf.float32, target_shape=(2, 2)),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8]], input_type=tf.float32,
             target_shape=(2, 2, 8)),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]], input_type=tf.float32,
             target_shape=(4, 24)),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]], input_type=tf.float32,
             target_shape=(32, 3, 2))]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_reshape(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                           use_new_frontend):
        self._test(*self.create_keras_reshape_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
