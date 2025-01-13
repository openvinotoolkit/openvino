# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(346546756)


class TestComplexAbs(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = rng.uniform(-2.0, 2.0, param_real_shape).astype(np.float32)
        inputs_data['param_imag:0'] = rng.uniform(-2.0, 2.0, param_imag_shape).astype(np.float32)
        return inputs_data

    def create_complex_abs_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)
            tf.raw_ops.ComplexAbs(x=complex)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[], [2], [1, 3], [2, 3, 4], [3, 4, 5, 6]])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_abs(self, input_shape, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_complex_abs_net(input_shape),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
