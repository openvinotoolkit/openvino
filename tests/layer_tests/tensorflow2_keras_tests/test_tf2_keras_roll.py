# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasRoll(CommonTF2LayerTest):
    def create_keras_roll_net(self, shift, axis, input_names, input_shapes, input_type, ir_version):
        # create TensorFlow 2 model with Roll operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        inputs = []
        for ind in range(len(input_names)):
            inputs.append(tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind]))
        y = tf.roll(inputs, shift=shift, axis=axis)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data = [
        dict(shift=[-20, 3, 0], axis=[0, 0, 0], input_names=["x1"], input_shapes=[[7]],
             input_type=tf.half),
        dict(shift=[1], axis=[-1], input_names=["x1"], input_shapes=[[4, 3]],
             input_type=tf.float32),
        dict(shift=[1, 5, -7], axis=[0, 1, 1], input_names=["x1"], input_shapes=[[2, 3, 5]],
             input_type=tf.float16),
        dict(shift=[11, -8], axis=[-1, -2], input_names=["x1"], input_shapes=[[3, 4, 3, 1]],
             input_type=tf.int32),
        dict(shift=[7, -2, 5], axis=[0, -1, -1], input_names=["x1"], input_shapes=[[5, 2, 3, 7]],
             input_type=tf.int64),
        dict(shift=[1, -2], axis=[0, 1], input_names=["x1"], input_shapes=[[2, 4, 3, 5]],
             input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_keras_roll(self, params, ie_device, precision, ir_version, temp_dir,
                        use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Roll is not supported on GPU")
        self._test(*self.create_keras_roll_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
