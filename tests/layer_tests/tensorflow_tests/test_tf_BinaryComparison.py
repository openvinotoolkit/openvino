# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import mix_array_with_several_values

rng = np.random.default_rng(23356)


class TestBinaryComparison(CommonTFLayerTest):
    def _generate_value(self, input_shape, input_type):
        if np.issubdtype(input_type, np.floating):
            gen_value = rng.uniform(-5.0, 5.0, input_shape).astype(input_type)
        elif np.issubdtype(input_type, np.signedinteger):
            gen_value = rng.integers(-8, 8, input_shape).astype(input_type)
        else:
            gen_value = rng.integers(8, 16, input_shape).astype(input_type)
        return gen_value

    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        input_type = self.input_type

        inputs_data = {}
        y_value = None
        if self.is_const:
            y_value = self.y_value
            y_shape = y_value.shape
        else:
            assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
            y_shape = inputs_info['y:0']
            y_value = self._generate_value(y_shape, input_type)
            inputs_data['y:0'] = y_value

        # generate x value so that some elements will be equal, less, greater than y value element-wise
        squeeze_dims = 0 if len(y_shape) <= len(x_shape) else len(y_shape) - len(x_shape)
        zeros_list = [0] * squeeze_dims
        y_value = y_value[tuple(zeros_list)]
        y_value_minus_one = y_value - 1
        y_value_plus_one = y_value + 1

        x_value = self._generate_value(x_shape, input_type)
        # mix input data with preferable values
        x_value = mix_array_with_several_values(x_value, [y_value, y_value_plus_one, y_value_minus_one], rng)
        inputs_data['x:0'] = x_value

        return inputs_data

    def create_binary_comparison_net(self, input_shape1, input_shape2, binary_op, is_const, input_type):
        compare_ops_map = {
            'Equal': tf.raw_ops.Equal,
            'NotEqual': tf.raw_ops.NotEqual,
            'Greater': tf.raw_ops.Greater,
            'GreaterEqual': tf.raw_ops.GreaterEqual,
            'Less': tf.raw_ops.Less,
            'LessEqual': tf.raw_ops.LessEqual,
        }

        self.input_type = input_type
        self.is_const = is_const
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape1, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape2, 'y')
            if is_const:
                self.y_value = self._generate_value(input_shape2, input_type)
                y = tf.constant(self.y_value, dtype=input_type)
            compare_ops_map[binary_op](x=x, y=y)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape1', [[], [4], [3, 4], [2, 3, 4]])
    @pytest.mark.parametrize('input_shape2', [[4], [3, 4]])
    @pytest.mark.parametrize('binary_op', ['Equal', 'NotEqual', 'Greater', 'GreaterEqual', 'Less', 'LessEqual'])
    @pytest.mark.parametrize('is_const', [False, True])
    @pytest.mark.parametrize('input_type', [np.int8, np.uint8, np.int16,
                                            np.int32, np.int64,
                                            np.float16, np.float32, np.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_binary_comparison(self, input_shape1, input_shape2, binary_op, is_const, input_type,
                               ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        if ie_device == 'GPU' and input_type == np.int16 and is_const:
            pytest.skip('150501: Accuracy error on GPU for int16 type and constant operand')
        self._test(*self.create_binary_comparison_net(input_shape1, input_shape2, binary_op, is_const, input_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
