# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestMatrixInverse(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        inputs_data = {}
        inputs_data['input:0'] = self._generate_invertible_matrices(self.input_shape)

        return inputs_data
    
    def _generate_invertible_matrices(self, input_shape):
        if input_shape == [2, 2]:
            return np.array([[1, 2],
                             [3, 1]
                            ], dtype=np.float32)
        elif input_shape == [3, 3]:
            return np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 1]
                            ], dtype=np.float32)
        elif input_shape == [4, 4]:
            return np.array([[1, 2, 3, 4],
                             [5, 6, 7, 8],
                             [9, 10, 2, 1],
                             [13, 14, 2, 1]
                            ], dtype=np.float32)
        elif input_shape == [2, 4, 4]:
            return np.array([[[10, 2, 3, 4],
                              [5, 10, 7, 8],
                              [9, 10, 11, 12],
                              [13, 14, 15, 16]],
                             [[1, 2, 3, 4],
                              [5, 6, 7, 8],
                              [9, 10, 6, 12],
                              [13, 14, 15, 10]]
                            ], dtype=np.float32)
                             
    def create_matrix_inverse_net(self, input_shape, adjoint):
        self.input_shape = input_shape
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_tensor = tf.compat.v1.placeholder(np.float32, input_shape, 'input')
            tf.raw_ops.MatrixInverse(input=input_tensor, adjoint=adjoint)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("input_shape", [[2, 2], [3, 3], [2, 4, 4]])
    @pytest.mark.parametrize("adjoint", [None, False, True])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_inverse_basic(self, input_shape, adjoint, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("GPU does not support Inverse operation")
        self._test(*self.create_matrix_inverse_net(input_shape=input_shape, adjoint=adjoint),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)