# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasActivityRegularization(CommonTF2LayerTest):
    def create_keras_activity_regularization_net(self, l1_param, l2_param, input_names,
                                                 input_shapes, input_type,
                                                 ir_version):
        # create TensorFlow 2 model with Keras ActivityRegularization operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:],
                            name=input_names[0])  # Variable-length sequence of ints
        y = tf.keras.layers.ActivityRegularization(l1=l1_param, l2=l2_param)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(l1_param=0.05, l2_param=0.08, input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
             input_type=tf.float32),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_activity_regularization_case1_float32(self, params, ie_device, precision,
                                                         ir_version, temp_dir,
                                                         use_legacy_frontend):
        pytest.skip("Error: failed due to missing a required argument: x1")
        self._test(*self.create_keras_activity_regularization_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_extended_float32 = [
        dict(l1_param=0.07, l2_param=0.05, input_names=["x1"], input_shapes=[[1]],
             input_type=tf.float32),
        dict(l1_param=0.05, l2_param=0.06, input_names=["x1"], input_shapes=[[4]],
             input_type=tf.float32),
        dict(l1_param=0.05, l2_param=0.05, input_names=["x1"], input_shapes=[[5, 4]],
             input_type=tf.float32),
        dict(l1_param=0.0, l2_param=0.05, input_names=["x1"], input_shapes=[[5, 4, 8]],
             input_type=tf.float32),
        dict(l1_param=0.05, l2_param=0.0, input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
             input_type=tf.float32),
        dict(l1_param=0.05, l2_param=0.08, input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]],
             input_type=tf.float32)
    ]

    @pytest.mark.parametrize("params", test_data_extended_float32)
    @pytest.mark.nightly
    def test_keras_activity_regularization_case_2_float32(self, params, ie_device, precision,
                                                          ir_version, temp_dir):
        self._test(*self.create_keras_activity_regularization_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   **params)
