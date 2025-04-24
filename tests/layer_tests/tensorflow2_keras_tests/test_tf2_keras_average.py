# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasAverage(CommonTF2LayerTest):
    def create_keras_average_net(self, input_names, input_shapes, input_type, ir_version):
        # create TensorFlow 2 model with Keras Average operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state

        inputs = []
        for ind in range(len(input_names)):
            inputs.append(tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind]))

        y = tf.keras.layers.Average()(inputs)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8], [5, 4, 8]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8, 2], [5, 4, 8, 2]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2", "x3"],
                              input_shapes=[[5, 4, 8], [5, 4, 8], [5, 4, 8]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2", "x3"],
                              input_shapes=[[5, 4, 8, 2], [5, 4, 8, 2], [5, 4, 8, 2]],
                              input_type=tf.float32),
                         ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_average_case1_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_legacy_frontend):
        self._test(*self.create_keras_average_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_extended_float32 = [dict(input_names=["x1", "x2"], input_shapes=[[1], [1]],
                                       input_type=tf.float32),
                                  dict(input_names=["x1", "x2"], input_shapes=[[5], [5]],
                                       input_type=tf.float32),
                                  dict(input_names=["x1", "x2"], input_shapes=[[5, 4], [5, 4]],
                                       input_type=tf.float32),
                                  dict(input_names=["x1", "x2"],
                                       input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
                                       input_type=tf.float32)
                                  ]

    @pytest.mark.parametrize("params", test_data_extended_float32)
    @pytest.mark.nightly
    def test_keras_average_case2_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                         use_legacy_frontend):
        self._test(*self.create_keras_average_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

    test_data_float32_several_inputs = [
        dict(input_names=["x1", "x2", "x3"], input_shapes=[[1], [1], [1]], input_type=tf.float32),
        dict(input_names=["x1", "x2", "x3"], input_shapes=[[4], [4], [4]], input_type=tf.float32),
        dict(input_names=["x1", "x2", "x3"], input_shapes=[[5, 4], [5, 4], [5, 4]],
             input_type=tf.float32),
        dict(input_names=["x1", "x2", "x3"],
             input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
             input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32_several_inputs)
    @pytest.mark.nightly
    def test_keras_average_float32_several_inputs(self, params, ie_device, precision, ir_version,
                                                  temp_dir, use_legacy_frontend):
        self._test(*self.create_keras_average_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
