# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(62362)

class TestExpandDims(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'input:0' in inputs_info, "Test error: inputs_info must contain `input`"
        x_shape = inputs_info['input:0']
        inputs_data = {}
        inputs_data['input:0'] = np.random.randint(-10, 10, x_shape).astype(np.float32)
        return inputs_data

    def create_expand_dims_net(self, input_shape, axis):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            axis = tf.constant(axis, dtype=tf.int32)
            tf.raw_ops.ExpandDims(input=input, axis=axis)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_basic = [
        dict(input_shape=[], axis=0),
        dict(input_shape=[2, 3], axis=1),
        dict(input_shape=[2, 3, 5], axis=-2),
    ]

    @pytest.mark.parametrize("params", test_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_expand_dims_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_expand_dims_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestExpandDimsComplex(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info

        input_shape = inputs_info['param_real:0']

        inputs_data = {}
        inputs_data['param_real:0'] = rng.integers(-10.0, 10.0, input_shape).astype(np.float32)
        inputs_data['param_imag:0'] = rng.integers(-10.0, 10.0, input_shape).astype(np.float32)

        return inputs_data

    def create_expand_dims_complex_net(self, axis_dtype, input_shape, axis):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')

            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            axis = tf.constant(axis, dtype=axis_dtype)

            result = tf.raw_ops.ExpandDims(input=complex, axis=axis)

            tf.raw_ops.Real(input=result)
            tf.raw_ops.Imag(input=result)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_basic = [
        dict(input_shape=[], axis=0),
        dict(input_shape=[2, 3], axis=1),
        dict(input_shape=[2, 3, 4], axis=-1),
        dict(input_shape=[2, 6, 5], axis=-2),
    ]

    @pytest.mark.parametrize("axis_dtype", [np.int32, np.int64])
    @pytest.mark.parametrize("op_args", test_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_expand_dims_basic_complex(self, axis_dtype, op_args, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_expand_dims_complex_net(axis_dtype, **op_args),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
