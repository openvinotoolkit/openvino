# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSlice(CommonTFLayerTest):
    def create_slice_net(self, input_shape, input_type, begin_value, size_value):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_x = tf.compat.v1.placeholder(input_type, input_shape, 'input_x')
            begin = tf.constant(begin_value, tf.int32)
            size = tf.constant(size_value, tf.int32)
            tf.raw_ops.Slice(input=input_x, begin=begin, size=size)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
        dict(input_shape=[6], input_type=tf.float32, begin_value=[2], size_value=[2]),
        dict(input_shape=[2, 5, 3], input_type=tf.int32, begin_value=[0, 1, 0], size_value=[-1, 1, -1]),
        dict(input_shape=[10, 5, 1, 5], input_type=tf.float32, begin_value=[5, 1, 0, 3], size_value=[2, 4, -1, -1]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_slice_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_slice_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestComplexSlice(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_slice_net(self, input_shape, begin_value, size_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(tf.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(tf.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)
            
            begin = tf.constant(begin_value, tf.int32)
            size = tf.constant(size_value, tf.int32)
            slice = tf.raw_ops.Slice(input=complex, begin=begin, size=size)
            tf.raw_ops.Real(input=slice)
            tf.raw_ops.Imag(input=slice)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[6], begin_value=[2], size_value=[2]),
        dict(input_shape=[2, 5, 3], begin_value=[0, 1, 0], size_value=[-1, 1, -1]),
        dict(input_shape=[10, 5, 1, 5], begin_value=[5, 1, 0, 3], size_value=[2, 4, -1, -1]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_slice(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(
            *self.create_complex_slice_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

