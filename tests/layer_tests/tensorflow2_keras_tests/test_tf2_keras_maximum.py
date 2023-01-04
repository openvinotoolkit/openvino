# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.layer_test_class import check_ir_version
from common.tf2_layer_test_class import CommonTF2LayerTest
from unit_tests.utils.graph import build_graph


class TestKerasMaximum(CommonTF2LayerTest):
    def create_keras_maximum_net(self, input_names, input_shapes, input_type, ir_version):
        """
               Tensorflow2 Keras net:                     IR net:
                      Input               =>               Input
                        |                                    |
                    Maximum                               Maximum
        """
        # create TensorFlow 2 model with Keras Add operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        inputs = []
        for ind in range(len(input_names)):
            inputs.append(tf.keras.Input(shape=input_shapes[ind][1:], name=input_names[ind]))
        y = tf.keras.layers.Maximum()(inputs)
        tf2_net = tf.keras.Model(inputs=inputs, outputs=[y])

        # create reference IR net
        ref_net = None

        op_name = "Maximum"
        if check_ir_version(10, None, ir_version):
            # convert NHWC to NCHW layout if tensor rank greater 3
            converted_input_shape = input_shapes[0].copy()
            if len(converted_input_shape) > 3:
                converted_input_shape[1] = input_shapes[0][-1]
                converted_input_shape[2:] = input_shapes[0][1:-1]
            if len(input_names) == 2:
                nodes_attributes = {
                    'input1': {'kind': 'op', 'type': 'Parameter'},
                    'input1_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'input2': {'kind': 'op', 'type': 'Parameter'},
                    'input2_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'op': {'kind': 'op', 'type': op_name},
                    'op_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'result': {'kind': 'op', 'type': 'Result'}
                }

                ref_net = build_graph(nodes_attributes,
                                      [('input1', 'input1_data'),
                                       ('input2', 'input2_data'),
                                       ('input1_data', 'op', {'in': 0}),
                                       ('input2_data', 'op', {'in': 1}),
                                       ('op', 'op_data'),
                                       ('op_data', 'result')
                                       ])
            elif len(input_names) == 3:
                nodes_attributes = {
                    'input1': {'kind': 'op', 'type': 'Parameter'},
                    'input1_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'input2': {'kind': 'op', 'type': 'Parameter'},
                    'input2_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'op1': {'kind': 'op', 'type': op_name},
                    'op1_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'input3': {'kind': 'op', 'type': 'Parameter'},
                    'input3_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'op2': {'kind': 'op', 'type': op_name},
                    'op2_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'result': {'kind': 'op', 'type': 'Result'}
                }

                ref_net = build_graph(nodes_attributes,
                                      [('input1', 'input1_data'),
                                       ('input2', 'input2_data'),
                                       ('input1_data', 'op1', {'in': 0}),
                                       ('input2_data', 'op1', {'in': 1}),
                                       ('op1', 'op1_data'),
                                       ('input3', 'input3_data'),
                                       ('op1_data', 'op2', {'in': 0}),
                                       ('input3_data', 'op2', {'in': 1}),
                                       ('op2', 'op2_data'),
                                       ('op2_data', 'result')
                                       ])
            else:
                AssertionError("Not supported case with input number greater 2")

        return tf2_net, ref_net

    test_data_float32_precommit = [
        dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
             input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    def test_keras_maximum_float32(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                                   use_new_frontend):
        self._test(*self.create_keras_maximum_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data_float32 = [dict(input_names=["x1", "x2"], input_shapes=[[5, 4], [5, 4]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8], [5, 4, 8]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2"], input_shapes=[[5, 4, 8, 2], [5, 4, 8, 2]],
                              input_type=tf.float32),
                         dict(input_names=["x1", "x2"],
                              input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
                              input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_maximum_float32(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                                   use_new_frontend):
        self._test(*self.create_keras_maximum_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data_float32_several_inputs_precommit = [
        dict(input_names=["x1", "x2", "x3"],
             input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
             input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32_several_inputs_precommit)
    @pytest.mark.precommit
    def test_keras_maximum_float32_several_inputs(self, params, ie_device, precision, ir_version,
                                                  temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_keras_maximum_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data_float32_several_inputs = [
        dict(input_names=["x1", "x2", "x3"],
             input_shapes=[[5, 4], [5, 4], [5, 4]],
             input_type=tf.float32),
        dict(input_names=["x1", "x2", "x3"],
             input_shapes=[[5, 4, 8], [5, 4, 8], [5, 4, 8]],
             input_type=tf.float32),
        dict(input_names=["x1", "x2", "x3"],
             input_shapes=[[5, 4, 8, 2], [5, 4, 8, 2], [5, 4, 8, 2]],
             input_type=tf.float32),
        dict(input_names=["x1", "x2", "x3"],
             input_shapes=[[5, 4, 8, 2, 3], [5, 4, 8, 2, 3], [5, 4, 8, 2, 3]],
             input_type=tf.float32)]

    @pytest.mark.parametrize("params", test_data_float32_several_inputs)
    @pytest.mark.nightly
    def test_keras_maximum_float32_several_inputs(self, params, ie_device, precision, ir_version,
                                                  temp_dir, use_old_api, use_new_frontend):
        self._test(*self.create_keras_maximum_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
