# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(253512)


class TestUnaryOpsAllRealDomain(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = rng.uniform(-5.0, 5.0, x_shape).astype(self.input_type)
        return inputs_data

    def create_unary_net(self, input_shape, input_type, op_type):
        op_type_map = {
            'Elu': lambda x: tf.raw_ops.Elu(features=x),
            'Sigmoid': tf.raw_ops.Sigmoid,
            'Sin': tf.raw_ops.Sin,
            'Sinh': tf.raw_ops.Sinh,
            'Cos': tf.raw_ops.Cos,
            'Cosh': tf.raw_ops.Cosh,
            'Exp': tf.raw_ops.Exp,
            'Atan': tf.raw_ops.Atan,
            'Softplus': lambda x: tf.raw_ops.Softplus(features=x),
            'Erf': tf.raw_ops.Erf,
            'Selu': lambda x: tf.raw_ops.Selu(features=x)
        }

        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            op_type_map[op_type](x=x)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("input_shape", [[], [2], [3, 4], [3, 2, 4]])
    @pytest.mark.parametrize("input_type", [np.float16, np.float32, np.float64])
    @pytest.mark.parametrize("op_type", ['Elu',
                                         'Sigmoid',
                                         'Sin',
                                         'Sinh',
                                         'Cos',
                                         'Cosh',
                                         'Exp',
                                         'Atan',
                                         'Softplus',
                                         'Erf',
                                         'Selu'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_unary_ops(self, input_shape, input_type, op_type,
                       ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        if platform.machine() in ["aarch64", "arm64", "ARM64"] and op_type in ['Cos', 'Cosh', 'Sinh', 'Exp']:
            pytest.skip("159585: accuracy error on ARM")
        self._test(*self.create_unary_net(input_shape, input_type, op_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=3 * 1e-3)
