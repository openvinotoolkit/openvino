# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


def x2_fun(x):
    return x + x


def x2_fun_shape(input_shape):
    return input_shape


def x1_plus_x2(inputs):
    x1 = inputs[0]
    x2 = inputs[1]
    return x1 + x2


def x1_plus_x2_shape(shapes):
    return shapes[0]


def x1_add_x2(inputs):
    return tf.keras.layers.Add()(inputs)


def x1_add_x2_shape(shapes):
    return shapes[0]


class TestKerasLambda(CommonTF2LayerTest):
    def create_keras_lambda_net(self, input_names, input_shapes, input_type, lmbd, exp_shapes,
                                ir_version):
        """
                create TensorFlow 2 model with Keras Lambda operation
        """
        lambda_operations = {
            "x2_fun": x2_fun,
            "x2_fun_shape": x2_fun_shape,
            "x1_plus_x2": x1_plus_x2,
            "x1_plus_x2_shape": x1_plus_x2_shape,
            "x1_add_x2": x1_add_x2,
            "x1_add_x2_shape": x1_add_x2_shape
        }

        lmbd = lambda_operations[lmbd]
        exp_shapes = lambda_operations[exp_shapes]

        tf.keras.backend.clear_session()  # For easy reset of notebook state

        inputs = [tf.keras.Input(shape=input_shapes[idx][1:], name=input_names[idx]) for idx in
                  range(len(input_shapes))]
        if len(inputs) > 1:
            y = tf.keras.layers.Lambda(function=lmbd, output_shape=exp_shapes)(inputs)
        else:
            y = tf.keras.layers.Lambda(function=lmbd, output_shape=exp_shapes)(inputs[0])

        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # TODO: add reference IR net. Now it is omitted since inference is more
        #  important and needs to be checked in the first
        ref_net = None
        return tf2_net, ref_net

    test_data_float32 = [
        dict(input_names=["x1"], input_shapes=[[5, 4, 3]], input_type=tf.float32,
             lmbd="x2_fun", exp_shapes="x2_fun_shape"),
        dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 3], [5, 4, 3]], input_type=tf.float32,
             lmbd="x1_plus_x2", exp_shapes="x1_plus_x2_shape"),
        dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 3], [5, 4, 3]], input_type=tf.float32,
             lmbd="x1_add_x2", exp_shapes="x1_add_x2_shape"),
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_keras_lambda_float32(self, params, ie_device, precision, temp_dir, ir_version, use_old_api,
                                  use_new_frontend):
        self._test(*self.create_keras_lambda_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_old_api=use_old_api, use_new_frontend=use_new_frontend, **params)
