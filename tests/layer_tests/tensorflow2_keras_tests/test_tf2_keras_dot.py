# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasDot(CommonTF2LayerTest):
    def create_keras_dot_net(self, input_names, input_shapes, input_type, axes, normalize,
                             ir_version):
        """
                create TensorFlow 2 model with Keras Dot operation
        """
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        inputs = [tf.keras.Input(shape=input_shapes[0][1:], name=input_names[0]),
                  tf.keras.Input(shape=input_shapes[1][1:], name=input_names[1])]
        y = tf.keras.layers.Dot(axes, normalize=normalize)(inputs)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None

        return tf2_net, ref_net

    test_data_normalize_float32 = [
        dict(input_names=["x", "y"], input_shapes=[[5, 4], [5, 4]], input_type=tf.float32, axes=1,
             normalize=False),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 4], [5, 1, 4]], input_type=tf.float32,
             axes=2,
             normalize=False),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 4], [5, 1, 4]], input_type=tf.float32,
             axes=1,
             normalize=False),
        dict(input_names=["x", "y"], input_shapes=[[5, 4], [5, 4]], input_type=tf.float32, axes=1,
             normalize=True),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 4], [5, 1, 4]], input_type=tf.float32,
             axes=1, normalize=True),
    ]

    @pytest.mark.parametrize("params", test_data_normalize_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_dot_normalize_float32(self, params, ie_device, precision, temp_dir, ir_version,
                                         use_old_api, use_new_frontend):
        self._test(*self.create_keras_dot_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_old_api=use_old_api, use_new_frontend=use_new_frontend, **params)

    test_data_difficult_axes_float32 = [
        dict(input_names=["x", "y"], input_shapes=[[5, 4, 4], [5, 4, 4]], input_type=tf.float32,
             axes=(1, 2),
             normalize=False),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 5, 2], [5, 1, 2, 5]],
             input_type=tf.float32, axes=(2, 3),
             normalize=False),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 64, 2], [5, 1, 16, 64]],
             input_type=tf.float32,
             axes=(2, 3), normalize=False),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 4], [5, 1, 4]], input_type=tf.float32,
             axes=-2, normalize=False),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 1024, 2], [5, 1, 8, 1024]],
             input_type=tf.float32,
             axes=(-2, -1), normalize=False),
    ]

    @pytest.mark.parametrize("params", test_data_difficult_axes_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_dot_difficult_axes_float32(self, params, ie_device, precision, temp_dir,
                                              ir_version, use_old_api, use_new_frontend):
        self._test(*self.create_keras_dot_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_old_api=use_old_api, use_new_frontend=use_new_frontend, **params)

    test_data_normalize_higher_rank = [
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 4], [5, 1, 4]], input_type=tf.float32,
             axes=2, normalize=True),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 1024, 2], [5, 1, 8, 1024]],
             input_type=tf.float32,
             axes=(2, 3), normalize=True),
        dict(input_names=["x", "y"], input_shapes=[[5, 4, 4], [5, 4, 4]], input_type=tf.float32,
             axes=(1, 2),
             normalize=True),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 5, 2], [5, 1, 2, 5]],
             input_type=tf.float32, axes=(2, 3),
             normalize=True),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 4], [5, 1, 4]], input_type=tf.float32,
             axes=-2, normalize=True),
        dict(input_names=["x", "y"], input_shapes=[[5, 1, 1024, 2], [5, 1, 4, 1024]],
             input_type=tf.float32,
             axes=(-2, -1), normalize=True),

    ]

    @pytest.mark.parametrize("params", test_data_normalize_higher_rank)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_dot_normalize_higher_rank(self, params, ie_device, precision, temp_dir,
                                             ir_version, use_old_api, use_new_frontend):
        self._test(*self.create_keras_dot_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_old_api=use_old_api, use_new_frontend=use_new_frontend, **params)
