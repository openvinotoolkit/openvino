# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestReshape(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'tensor:0' in inputs_info
        tensor_shape = inputs_info['tensor:0']
        inputs_data = {}
        inputs_data['tensor:0'] = np.random.randint(-10, 10, tensor_shape).astype(self.input_type)

        return inputs_data

    def create_reshape_net(self, input_shape, input_type, target_shape):
        self.input_type = input_type
        types_map = {
            np.float32: tf.float32,
            np.int32: tf.int32
        }
        assert input_type in types_map, "Incorrect test case"
        tf_type = types_map[input_type]
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tensor = tf.compat.v1.placeholder(tf_type, input_shape, 'tensor')
            shape = tf.constant(target_shape, dtype=tf.int32)
            tf.raw_ops.Reshape(tensor=tensor, shape=shape)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 6], input_type=np.float32, target_shape=[2, 3, 2]),
        dict(input_shape=[2, 4, 5], input_type=np.int32, target_shape=[4, -1, 5]),
        dict(input_shape=[1], input_type=np.float32, target_shape=[]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_reshape_basic(self, params, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        if ie_device == 'GPU' and params['target_shape'] == []:
            pytest.skip("timeout issue on GPU")
        self._test(*self.create_reshape_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestComplexReshape(CommonTFLayerTest):
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

    def create_complex_transpose_net(self, input_shape, target_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            transpose = tf.raw_ops.Reshape(tensor=complex, shape=target_shape)
            tf.raw_ops.Real(input=transpose)
            tf.raw_ops.Imag(input=transpose)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 6], target_shape=[2, 3, 2]),
        dict(input_shape=[2, 4, 5], target_shape=[4, -1, 5]),
        dict(input_shape=[1], target_shape=[])
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_reshape(self, params, ie_device, precision, ir_version, temp_dir,
                             use_legacy_frontend):
        self._test(
            *self.create_complex_transpose_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_legacy_frontend=use_legacy_frontend)
