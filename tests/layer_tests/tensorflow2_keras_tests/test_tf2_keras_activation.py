# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class TestKerasActivation(CommonTF2LayerTest):
    def create_keras_activation_net(self, activation_func, input_names, input_shapes, input_type,
                                    ir_version):
        activation_func_structure = {
            # pytest-xdist can't execute the tests in parallel because workers can't compare tests scopes before run
            # tf.nn.<activation> operation have no "==" operation to be compared
            "elu": tf.nn.elu,
            "relu": tf.nn.relu,
            "sigmoid": tf.nn.sigmoid,
            "softmax": tf.nn.softmax,
            "softsign": tf.nn.softsign,
            "swish": tf.nn.swish,
            "tanh": tf.nn.tanh,
            "softplus": tf.nn.softplus,
            "selu": tf.nn.selu
        }
        activation_func = activation_func_structure[activation_func]

        # create TensorFlow 2 model with Keras Activation operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:],
                            name=input_names[0])  # Variable-length sequence of ints
        y = tf.keras.layers.Activation(activation_func)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data_float32 = [
        dict(activation_func="elu", input_names=["x1"], input_shapes=[[5, 4]],
             input_type=tf.float32),
        dict(activation_func="relu", input_names=["x1"], input_shapes=[[5, 4, 8]],
             input_type=tf.float32),
        dict(activation_func="sigmoid", input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
             input_type=tf.float32),
        pytest.param(dict(activation_func="softmax", input_names=["x1"], input_shapes=[[5, 4, 8]],
                          input_type=tf.float32), marks=pytest.mark.precommit_tf_fe),
        dict(activation_func="softsign", input_names=["x1"], input_shapes=[[5, 4]],
             input_type=tf.float32),
        dict(activation_func="swish", input_names=["x1"], input_shapes=[[5, 4, 8]],
             input_type=tf.float32),
        dict(activation_func="tanh", input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
             input_type=tf.float32),
        dict(activation_func="softsign", input_names=["x1"], input_shapes=[[5]],
             input_type=tf.float32),
        dict(activation_func="relu", input_names=["x1"], input_shapes=[[1]], input_type=tf.float32),
        dict(activation_func="swish", input_names=["x1"], input_shapes=[[5, 4, 8, 3, 4]],
             input_type=tf.float32),

        pytest.param(dict(activation_func="softplus", input_names=["x1"], input_shapes=[[5, 7, 6]],
                          input_type=tf.float32), marks=pytest.mark.xfail(reason="49516")),
        pytest.param(dict(activation_func="selu", input_names=["x1"], input_shapes=[[5, 7, 6]],
                          input_type=tf.float32), marks=[pytest.mark.xfail(reason="49512"),
                                                         pytest.mark.skip(
                                                             reason="Selu is unsupported in MO")])
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_activation_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_old_api, use_new_frontend):
        self._test(*self.create_keras_activation_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version, use_old_api=use_old_api,
                   use_new_frontend=use_new_frontend, **params)


class TestKerasMultipleOutputsWithActivation(CommonTF2LayerTest):
    def create_keras_multiple_outputs_net(self, input_names, input_shapes, output_names, ir_version):
        # create TensorFlow 2 model with multiple outputs
        tf.keras.backend.clear_session()  # For easy reset of notebook state

        inputs = []
        outputs = []
        for ind in range(len(input_names)):
            input = tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind])
            inputs.append(input)
            outputs.append(tf.keras.layers.Activation(tf.nn.sigmoid)(input))
        tf2_net = tf.keras.Model(inputs=inputs, outputs=outputs)

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_data = [dict(input_names=["k", "b", "m", "c", "x"],
                      input_shapes=[[1, 1], [1, 3], [1, 2], [1, 5], [1, 4]],
                      output_names=["Func/PartitionedCall/output/_5:0",
                                    "Func/PartitionedCall/output/_6:0",
                                    "Func/PartitionedCall/output/_7:0",
                                    "Func/PartitionedCall/output/_8:0",
                                    "Func/PartitionedCall/output/_9:0"])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_outputs_order_test(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_old_api, use_new_frontend):
        self._test(*self.create_keras_multiple_outputs_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version, use_old_api=use_old_api,
                   use_new_frontend=use_new_frontend, **params)
