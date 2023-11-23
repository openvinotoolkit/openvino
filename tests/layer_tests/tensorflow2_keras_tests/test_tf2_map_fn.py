# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf

from common.tf2_layer_test_class import CommonTF2LayerTest


class MapFNLayer(tf.keras.layers.Layer):
    def __init__(self, fn, input_type, fn_output_signature, back_prop):
        super(MapFNLayer, self).__init__()
        self.fn = fn
        self.input_type = input_type
        self.fn_output_signature = fn_output_signature
        self.back_prop = back_prop

    def call(self, x):
        return tf.map_fn(self.fn, x, dtype=self.input_type,
                         fn_output_signature=self.fn_output_signature,
                         back_prop=self.back_prop)


class TestMapFN(CommonTF2LayerTest):
    def create_map_fn_net(self, fn, input_type, fn_output_signature, back_prop,
                          input_names, input_shapes, ir_version):
        # create TensorFlow 2 model using MapFN construction
        tf.keras.backend.clear_session()
        inputs = []
        for ind in range(len(input_names)):
            inputs.append(tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind],
                                         dtype=input_type))
        map_fn_layer = MapFNLayer(fn, input_type, fn_output_signature, back_prop)(inputs)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[map_fn_layer])

        # TODO: add reference IR net. Now it is omitted and tests only inference result that is more important
        ref_net = None

        return tf2_net, ref_net

    test_basic = [
        dict(fn=lambda x: x[0] * x[1] + x[2], input_type=tf.float32,
             fn_output_signature=tf.float32, back_prop=False,
             input_names=["x1", "x2", "x3"], input_shapes=[[2, 3, 4], [2, 3, 4], [2, 3, 4]]),
        pytest.param(dict(fn=lambda x: (x[0] + x[1] + x[2], x[0] - x[2] + x[1], 2 + x[2]),
                          input_type=tf.float32,
                          fn_output_signature=(tf.float32, tf.float32, tf.float32), back_prop=True,
                          input_names=["x1", "x2", "x3"],
                          input_shapes=[[2, 1, 3, 4], [2, 1, 3, 4], [2, 1, 3, 4]]),
                     marks=pytest.mark.xfail(reason="61587"))
    ]

    @pytest.mark.parametrize("params", test_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_basic(self, params, ie_device, precision, ir_version, temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_old_api=use_old_api, use_new_frontend=use_new_frontend,
                   **params)

    test_multiple_inputs = [
        dict(fn=lambda x: x[0] * x[1], input_type=tf.float32,
             fn_output_signature=tf.float32, back_prop=True,
             input_names=["x1", "x2"], input_shapes=[[2, 4], [2, 4]]),
        dict(fn=lambda x: x[0] * x[1] + 2 * x[2], input_type=tf.float32,
             fn_output_signature=tf.float32, back_prop=False,
             input_names=["x1", "x2", "x3"], input_shapes=[[2, 1, 3, 4],
                                                           [2, 1, 3, 4],
                                                           [2, 1, 3, 4]])
    ]

    @pytest.mark.parametrize("params", test_multiple_inputs)
    @pytest.mark.nightly
    def test_multiple_inputs(self, params, ie_device, precision, ir_version, temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_old_api=use_old_api, use_new_frontend=use_new_frontend,
                   **params)

    test_multiple_outputs = [
        pytest.param(dict(fn=lambda x: (x[0] * x[1], x[0] + x[1]), input_type=tf.float32,
                          fn_output_signature=(tf.float32, tf.float32), back_prop=True,
                          input_names=["x1", "x2"], input_shapes=[[2, 4], [2, 4]]),
                     marks=pytest.mark.xfail(reason="61587")),
        pytest.param(dict(fn=lambda x: (x[0] * x[1] + x[2], x[0] + x[2] * x[1], 2 * x[2]),
                          input_type=tf.float32,
                          fn_output_signature=(tf.float32, tf.float32, tf.float32), back_prop=True,
                          input_names=["x1", "x2", "x3"],
                          input_shapes=[[2, 1, 3], [2, 1, 3], [2, 1, 3]]),
                     marks=pytest.mark.xfail(reason="61587"))
    ]

    @pytest.mark.parametrize("params", test_multiple_outputs)
    @pytest.mark.nightly
    def test_multiple_outputs(self, params, ie_device, precision, ir_version, temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_old_api=use_old_api, use_new_frontend=use_new_frontend,
                   **params)

    test_multiple_inputs_outputs_int32 = [
        (lambda x: x[0] * x[1] + x[2], tf.int32, tf.int32, True, ["x1", "x2", "x3"], [[2, 1, 3], [2, 1, 3], [2, 1, 3]]),
        pytest.param(lambda x: (x[0] + x[1] + x[2], x[0] - x[2] + x[1], 2 + x[2]), tf.int32, (tf.int32, tf.int32, tf.int32), True, ["x1", "x2", "x3"], [[2, 1, 3, 4], [2, 1, 3, 4], [2, 1, 3, 4]],
                     marks=[pytest.mark.xfail(reason="61587"), pytest.mark.precommit_tf_fe])
    ]

    @pytest.mark.parametrize("fn, input_type, fn_output_signature, back_prop, input_names, input_shapes", test_multiple_inputs_outputs_int32)
    @pytest.mark.nightly
    def test_multiple_inputs_outputs_int32(self, fn, input_type, fn_output_signature, back_prop, input_names, input_shapes, ie_device, precision, ir_version, temp_dir,
                                           use_old_api, use_new_frontend):
        params = {"fn": fn, "input_type": input_type, "fn_output_signature": fn_output_signature, "back_prop": back_prop, "input_names": input_names, "input_shapes": input_shapes}
        self._test(*self.create_map_fn_net(**params, ir_version=ir_version), ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_old_api=use_old_api, use_new_frontend=use_new_frontend,
                   **params)
