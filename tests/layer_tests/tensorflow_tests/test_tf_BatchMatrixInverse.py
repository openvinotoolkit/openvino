# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestBatchMatrixInverse(CommonTFLayerTest):
    def prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = self._generate_invertible_matrices(x_shape)
        return inputs_data
    
    def _generate_invertible_matrices(self, input_shape):
        if input_shape == [2, 3, 3]:
            return np.array([
                [[1.0, 2.0, 3.0],
                 [0.0, 1.0, 4.0],
                 [5.0, 6.0, 0.0]],
                [[2.0, 0.0, 0.0],
                 [3.0, 4.0, 0.0],
                 [5.0, 6.0, 7.0]]
            ])
        elif input_shape == [2, 4, 4]:
            return np.array([
                [[1.0, 2.0, 3.0, 4.0],
                 [0.0, 5.0, 6.0, 7.0],
                 [0.0, 0.0, 8.0, 9.0],
                 [0.0, 0.0, 0.0, 10.0]],
                [[2.0, 0.0, 0.0, 0.0],
                 [3.0, 4.0, 0.0, 0.0],
                 [5.0, 6.0, 7.0, 0.0],
                 [8.0, 9.0, 10.0, 11.0]]
            ])
        elif input_shape == [5, 2, 2]:
            return np.array([
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 0.0], [3.0, 4.0]],
                [[5.0, 6.0], [0.0, 7.0]],
                [[8.0, 9.0], [10.0, 0.0]],
                [[0.0, 11.0], [12.0, 13.0]]
            ])
        else:
            raise ValueError(f"Unsupported input shape: {input_shape}")

    def create_batch_matrix_inverse_net(self, input_shape, data_type, adjoint):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(data_type, input_shape, 'x')
            y = tf.raw_ops.BatchMatrixInverse(input=x, adjoint=adjoint)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        return tf_net, None

    @pytest.mark.parametrize("input_shape", [
        [2, 3, 3],
        [2, 4, 4],
        [5, 2, 2],
    ])
    @pytest.mark.parametrize("input_type", [
        np.float32,
        np.float64
    ])
    @pytest.mark.parametrize("adjoint", [
        False,
        True,
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_batch_matrix_inverse_basic(self, input_shape, input_type, adjoint, 
                                        ie_device, precision, ir_version, 
                                        temp_dir, use_legacy_frontend):
        self._test(*self.create_batch_matrix_inverse_net(
            input_shape=input_shape,
            data_type=tf.as_dtype(input_type),
            adjoint=adjoint
        ), ie_device, precision, ir_version, temp_dir=temp_dir,
           use_legacy_frontend=use_legacy_frontend)
