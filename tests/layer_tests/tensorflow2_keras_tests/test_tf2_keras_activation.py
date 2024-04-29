# Copyright (C) 2022-2024 Intel Corporation
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
            "gelu": tf.nn.gelu,
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
        dict(activation_func="gelu", input_names=["x1"], input_shapes=[[5, 4]],
             input_type=tf.float32),
        dict(activation_func="relu", input_names=["x1"], input_shapes=[[5, 4, 8]],
             input_type=tf.float32),
        dict(activation_func="sigmoid", input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
             input_type=tf.float32),
        dict(activation_func="softmax", input_names=["x1"], input_shapes=[[5, 4, 8]],
             input_type=tf.float32),
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
        dict(activation_func="softplus", input_names=["x1"], input_shapes=[[5, 7, 6]],
             input_type=tf.float32),
        pytest.param(dict(activation_func="selu", input_names=["x1"], input_shapes=[[5, 7, 6]],
                          input_type=tf.float32), marks=[pytest.mark.xfail(reason="49512"),
                                                         pytest.mark.skip(
                                                             reason="Selu is unsupported in MO")])
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_activation_float32(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        if params['activation_func'] == "swish":
            pytest.skip("Error: failed due to missing a required argument: x1")
        self._test(*self.create_keras_activation_net(**params, ir_version=ir_version), ie_device,
                   precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
