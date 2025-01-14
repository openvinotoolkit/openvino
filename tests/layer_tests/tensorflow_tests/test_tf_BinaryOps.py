# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


def generate_input(x_shape, x_type, bound=False):
    # usual function domain
    lower = -25
    upper = 25

    # specific domains
    if bound:
        lower = 1
        upper = 6

    if np.issubdtype(x_type, np.floating):
        return rng.uniform(lower, upper, x_shape).astype(x_type)
    elif np.issubdtype(x_type, np.signedinteger):
        return rng.integers(lower, upper, x_shape).astype(x_type)
    elif np.issubdtype(x_type, bool):
        return rng.integers(0, 2, x_shape).astype(x_type)

    return rng.uniform(lower, upper, x_shape).astype(x_type)


class TestBinaryOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        x_type = self.input_type

        inputs_data = {}
        inputs_data['x:0'] = generate_input(x_shape, x_type)
        return inputs_data

    def create_add_placeholder_const_net(self, x_shape, y_shape, op_type):
        import tensorflow as tf

        op_type_to_tf = {
            'Add': tf.raw_ops.Add,
            'AddV2': tf.raw_ops.AddV2,
            'Sub': tf.raw_ops.Sub,
            'Mul': tf.raw_ops.Mul,
            'Div': tf.raw_ops.Div,
            'RealDiv': tf.raw_ops.RealDiv,
            'SquaredDifference': tf.raw_ops.SquaredDifference,
            'Pow': tf.raw_ops.Pow,
            'Maximum': tf.raw_ops.Maximum,
            'Minimum': tf.raw_ops.Minimum,
            'Mod': tf.raw_ops.Mod,
            'FloorMod': tf.raw_ops.FloorMod,
            'FloorDiv': tf.raw_ops.FloorDiv,
            'Xdivy': tf.raw_ops.Xdivy,
        }

        input_type = np.float32
        if op_type in ['Pow']:
            input_type = np.int32
        self.input_type = input_type

        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, x_shape, 'x')
            bound = True if op_type in ['Pow', 'Div', 'Xdivy', 'RealDiv', 'Mod', 'FloorMod',
                                        'FloorDiv'] else False
            constant_value = generate_input(y_shape, input_type, bound)
            y = tf.constant(constant_value, dtype=input_type)
            op_type_to_tf[op_type](x=x, y=y, name=op_type)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('x_shape', [[2, 3, 4], [1, 2, 3, 4]])
    @pytest.mark.parametrize('y_shape', [[4], [2, 3, 4]])
    @pytest.mark.parametrize("op_type",
                             ['Add', 'AddV2', 'Sub', 'Mul', 'Div', 'RealDiv', 'SquaredDifference', 'Pow',
                              'Maximum', 'Minimum', 'Mod', 'FloorMod', 'FloorDiv', 'Xdivy'])
    @pytest.mark.nightly
    @pytest.mark.precommit
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_binary_op(self, x_shape, y_shape, ie_device, precision, ir_version, temp_dir, op_type,
                       use_legacy_frontend):
        if use_legacy_frontend and op_type in ['Xdivy']:
            pytest.skip("Xdivy op is supported only by new TF FE.")
        if op_type in ['Pow', 'Mod'] and ie_device == 'GPU':
            pytest.skip("For Mod and Pow GPU has inference mismatch")
        if op_type in ['Mod', 'FloorDiv', 'FloorMod']:
            pytest.skip("Inference mismatch for Mod and FloorDiv")
        self._test(*self.create_add_placeholder_const_net(x_shape=x_shape, y_shape=y_shape, op_type=op_type), ie_device,
                   precision, ir_version, temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)
