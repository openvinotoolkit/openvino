# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(21097)

op_type_to_tf = {
    'BitwiseAnd': tf.raw_ops.BitwiseAnd,
    'BitwiseOr': tf.raw_ops.BitwiseOr,
    'BitwiseXor': tf.raw_ops.BitwiseXor,
}


def generate_input(x_shape, x_type):
    if np.issubdtype(x_type, np.signedinteger):
        return rng.integers(-100, 100, x_shape).astype(x_type)
    return rng.integers(0, 200, x_shape).astype(x_type)


class TestBitwise(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}

        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data['x:0'] = generate_input(x_shape, self.input_type)
        if not self.is_y_const:
            assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
            y_shape = inputs_info['y:0']
            inputs_data['y:0'] = generate_input(y_shape, self.input_type)
        return inputs_data

    def create_bitwise_net(self, x_shape, y_shape, is_y_const, input_type, op_type):
        self.is_y_const = is_y_const
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, x_shape, 'x')
            if is_y_const:
                constant_value = generate_input(y_shape, input_type)
                y = tf.constant(constant_value, dtype=input_type)
            else:
                y = tf.compat.v1.placeholder(input_type, y_shape, 'y')
            op_type_to_tf[op_type](x=x, y=y, name=op_type)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('x_shape', [[4], [3, 4], [1, 2, 3, 4]])
    @pytest.mark.parametrize('y_shape', [[4], [2, 3, 4]])
    @pytest.mark.parametrize('is_y_const', [True, False])
    @pytest.mark.parametrize('input_type', [np.int8, np.int16, np.int32, np.int64,
                                            np.uint8, np.uint16, np.uint32, np.uint64])
    @pytest.mark.parametrize("op_type", ['BitwiseAnd', 'BitwiseOr', 'BitwiseXor'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_bitwise(self, x_shape, y_shape, is_y_const, input_type, op_type, ie_device, precision, ir_version,
                     temp_dir, use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("148540: Bitwise ops are not supported on GPU")
        self._test(*self.create_bitwise_net(x_shape=x_shape, y_shape=y_shape, is_y_const=is_y_const,
                                            input_type=input_type, op_type=op_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)
