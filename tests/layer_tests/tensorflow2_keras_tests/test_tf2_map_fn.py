# Copyright (C) 2022-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf2_layer_test_class import CommonTF2LayerTest

rng = np.random.default_rng()


def fn_1(x):
    return (x[0] * x[1] + x[2])


def fn_2(x):
    return (x[0] + x[1] + x[2], x[0] - x[2] + x[1], 2 + x[2])


def fn_3(x):
    return (x[0] * x[1])


def fn_4(x):
    return (x[0] * x[1] + 2 * x[2])


def fn_5(x):
    return (x[0] * x[1], x[0] + x[1])


def fn_6(x):
    return (x[0] * x[1] + x[2], x[0] + x[2] * x[1], 2 * x[2])


def fn_7(x):
    return (x[0] * x[1] + x[2])


def fn_8(x):
    return (x[0] + x[1] + x[2], x[0] - x[2] + x[1], 2 + x[2])


list_fns = [fn_1, fn_2, fn_3, fn_4, fn_5, fn_6, fn_7, fn_8]


class MapFNLayer(tf.keras.layers.Layer):
    def __init__(self, fn, input_type, fn_output_signature, back_prop):
        super(MapFNLayer, self).__init__()
        self.fn = list_fns[fn - 1]
        self.input_type = input_type
        self.fn_output_signature = fn_output_signature
        self.back_prop = back_prop

    def call(self, x):
        return tf.map_fn(self.fn, x, dtype=self.input_type,
                         fn_output_signature=self.fn_output_signature,
                         back_prop=self.back_prop)


class TestMapFN(CommonTF2LayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}
        for input_name in list(inputs_info.keys()):
            input_shape = inputs_info[input_name]
            if np.issubdtype(self.input_type, np.floating):
                inputs_data[input_name] = rng.uniform(-5.0, 5.0, input_shape).astype(self.input_type)
            else:
                inputs_data[input_name] = rng.integers(-8, 8, input_shape).astype(self.input_type)

        return inputs_data

    def create_map_fn_net(self, fn, input_type, fn_output_signature, back_prop,
                          input_names, input_shapes, ir_version):
        self.input_type = input_type.as_numpy_dtype
        # create TensorFlow 2 model using MapFN construction
        tf.keras.backend.clear_session()
        inputs = []
        lambdas = []
        for ind in range(len(input_names)):
            input_ = tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind],
                                    dtype=input_type)
            inputs.append(input_)
            # add additional layer Add to avoid
            # double tensor name for Parameter after EliminateLoopInputsOutputs elimination
            lambda_ = tf.keras.layers.Lambda(lambda x: x + 1)(input_)
            lambdas.append(lambda_)
        map_fn_layer = MapFNLayer(fn, input_type, fn_output_signature, back_prop)(lambdas)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[map_fn_layer])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_basic = [
        dict(fn=1, input_type=tf.float32,
             fn_output_signature=tf.float32, back_prop=False,
             input_names=["x1", "x2", "x3"], input_shapes=[[2, 3, 4], [2, 3, 4], [2, 3, 4]]),
        dict(fn=2,
             input_type=tf.float32,
             fn_output_signature=(tf.float32, tf.float32, tf.float32), back_prop=True,
             input_names=["x1", "x2", "x3"],
             input_shapes=[[2, 1, 3, 4], [2, 1, 3, 4], [2, 1, 3, 4]]),
    ]

    @pytest.mark.parametrize("params", test_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)

    test_multiple_inputs = [
        dict(fn=3, input_type=tf.float32,
             fn_output_signature=tf.float32, back_prop=True,
             input_names=["x1", "x2"], input_shapes=[[2, 4], [2, 4]]),
        dict(fn=4, input_type=tf.float32,
             fn_output_signature=tf.float32, back_prop=False,
             input_names=["x1", "x2", "x3"], input_shapes=[[2, 1, 3, 4],
                                                           [2, 1, 3, 4],
                                                           [2, 1, 3, 4]])
    ]

    @pytest.mark.parametrize("params", test_multiple_inputs)
    @pytest.mark.nightly
    def test_multiple_inputs(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)

    test_multiple_outputs = [
        pytest.param(dict(fn=5, input_type=tf.float32,
                          fn_output_signature=(tf.float32, tf.float32), back_prop=True,
                          input_names=["x1", "x2"], input_shapes=[[2, 4], [2, 4]]),
                     marks=pytest.mark.xfail(reason="61587")),
        pytest.param(dict(fn=6,
                          input_type=tf.float32,
                          fn_output_signature=(tf.float32, tf.float32, tf.float32), back_prop=True,
                          input_names=["x1", "x2", "x3"],
                          input_shapes=[[2, 1, 3], [2, 1, 3], [2, 1, 3]]),
                     marks=pytest.mark.xfail(reason="61587"))
    ]

    @pytest.mark.parametrize("params", test_multiple_outputs)
    @pytest.mark.nightly
    def test_multiple_outputs(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)

    test_multiple_inputs_outputs_int32 = [
        dict(fn=7,
             input_type=tf.int32,
             fn_output_signature=tf.int32, back_prop=True,
             input_names=["x1", "x2", "x3"],
             input_shapes=[[2, 1, 3], [2, 1, 3], [2, 1, 3]]),
        dict(fn=8,
             input_type=tf.int32,
             fn_output_signature=(tf.int32, tf.int32, tf.int32), back_prop=True,
             input_names=["x1", "x2", "x3"],
             input_shapes=[[2, 1, 3, 4], [2, 1, 3, 4], [2, 1, 3, 4]]),
    ]

    @pytest.mark.parametrize("params", test_multiple_inputs_outputs_int32)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_multiple_inputs_outputs_int32(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)
