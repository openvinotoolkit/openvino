# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.layer_test_class import check_ir_version
from common.tf2_layer_test_class import CommonTF2LayerTest
from unit_tests.utils.graph import build_graph


class TestKerasELU(CommonTF2LayerTest):
    def create_keras_elu_net(self, input_names, input_shapes, input_type, alpha, ir_version):
        """
               Tensorflow2 Keras net:                     IR net:
                      Input               =>               Input
                        |                                    |
                       ELU                                  Elu
        """
        # create TensorFlow 2 model with Keras ELU operation
        tf.keras.backend.clear_session()  # For easy reset of notebook state
        x1 = tf.keras.Input(shape=input_shapes[0][1:],
                            name=input_names[0])  # Variable-length sequence of ints
        y = tf.keras.layers.ELU(alpha=alpha)(x1)
        tf2_net = tf.keras.Model(inputs=[x1], outputs=[y])

        # create reference IR net
        ref_net = None

        if check_ir_version(10, None, ir_version):
            # convert NHWC to NCHW layout if tensor rank greater 3
            converted_input_shape = input_shapes[0].copy()
            if len(converted_input_shape) > 3:
                converted_input_shape[1] = input_shapes[0][-1]
                converted_input_shape[2:] = input_shapes[0][1:-1]
            if alpha == 1.0:
                nodes_attributes = {
                    'input1': {'kind': 'op', 'type': 'Parameter'},
                    'input1_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'elu': {'kind': 'op', 'type': 'Elu'},
                    'elu_data': {'shape': converted_input_shape, 'kind': 'data'},
                    'result': {'kind': 'op', 'type': 'Result'}
                }

                ref_net = build_graph(nodes_attributes,
                                      [('input1', 'input1_data'),
                                       ('input1_data', 'elu', {'in': 0}),
                                       ('elu', 'elu_data'),
                                       ('elu_data', 'result')])
            else:
                nodes_attributes = {
                    'input1': {'kind': 'op', 'type': 'Parameter'},
                    'input1_data': {'shape': converted_input_shape, 'kind': 'data'},

                    'alpha_input_data': {'kind': 'data', 'shape': [1], 'value': [0.0]},
                    'alpha': {'kind': 'op', 'type': 'Const'},
                    'alpha_data': {'kind': 'data'},

                    'const_input_data': {'kind': 'data', 'shape': [1], 'value': [alpha]},
                    'const': {'kind': 'op', 'type': 'Const'},
                    'const_data': {'kind': 'data'},

                    'greater': {'kind': 'op', 'type': 'Greater'},
                    'greater_data': {'shape': converted_input_shape, 'kind': 'data'},

                    'elu': {'kind': 'op', 'type': 'Elu'},
                    'elu_data': {'shape': converted_input_shape, 'kind': 'data'},

                    '1select': {'kind': 'op', 'type': 'Select'},
                    'select_data': {'shape': converted_input_shape, 'kind': 'data'},

                    '2multiply': {'kind': 'op', 'type': 'Multiply'},
                    'multiply_data': {'shape': converted_input_shape, 'kind': 'data'},

                    'result': {'kind': 'op', 'type': 'Result'}
                }

                ref_net = build_graph(nodes_attributes,
                                      [('input1', 'input1_data'),
                                       ('alpha_input_data', 'alpha'),
                                       ('alpha', 'alpha_data'),
                                       ('const_input_data', 'const'),
                                       ('const', 'const_data'),

                                       ('input1_data', 'greater', {'in': 0}),
                                       ('alpha_data', 'greater', {'in': 1}),
                                       ('greater', 'greater_data'),

                                       ('input1_data', 'elu', {'in': 0}),
                                       ('elu', 'elu_data'),

                                       ('const_data', '2multiply', {'in': 0}),
                                       ('elu_data', '2multiply', {'in': 1}),
                                       ('2multiply', 'multiply_data'),

                                       ('greater_data', '1select', {'in': 0}),
                                       ('elu_data', '1select', {'in': 1}),
                                       ('multiply_data', '1select', {'in': 2}),
                                       ('1select', 'select_data'),

                                       ('select_data', 'result')])

        return tf2_net, ref_net

    test_data_float32_precommit = [dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]],
                                        input_type=tf.float32, alpha=1.0)]

    @pytest.mark.parametrize("params", test_data_float32_precommit)
    @pytest.mark.precommit
    def test_keras_elu_float32(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                               use_new_frontend):
        self._test(*self.create_keras_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data_float32 = [
        dict(input_names=["x1"], input_shapes=[[5, 4]], input_type=tf.float32, alpha=1.0),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8]], input_type=tf.float32, alpha=1.0),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]], input_type=tf.float32, alpha=1.0),
        dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]], input_type=tf.float32, alpha=1.0)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_keras_elu_float32(self, params, ie_device, precision, ir_version, temp_dir, use_old_api,
                               use_new_frontend):
        self._test(*self.create_keras_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)

    test_data_float32_alpha2 = [dict(input_names=["x1"], input_shapes=[[5, 4]],
                                     input_type=tf.float32, alpha=2.0),
                                dict(input_names=["x1"], input_shapes=[[5, 4, 8]],
                                     input_type=tf.float32, alpha=3.0),
                                dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3]],
                                     input_type=tf.float32, alpha=4.0),
                                dict(input_names=["x1"], input_shapes=[[5, 4, 8, 3, 2]],
                                     input_type=tf.float32, alpha=5.0)]

    @pytest.mark.parametrize("params", test_data_float32_alpha2)
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(reason="51109")
    def test_keras_elu_float32_alpha2(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_old_api, use_new_frontend):
        self._test(*self.create_keras_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, temp_dir=temp_dir, use_old_api=use_old_api, ir_version=ir_version,
                   use_new_frontend=use_new_frontend, **params)
