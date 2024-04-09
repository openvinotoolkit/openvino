# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestMatrixInverse(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input:0']
        inputs_data = {}

        self.invertible_matrices = self._generate_invertible_matrices()

        if len(input_shape) == 3 and input_shape[1] == input_shape[2]:  # Square matrices with batches
            matrices = np.random.choice(self.invertible_matrices, size=input_shape[0])
            inputs_data['input:0'] = np.stack(matrices, axis=0)
        else:
            inputs_data['input:0'] = np.random.choice(self.invertible_matrices)

        return inputs_data
    
    def _generate_invertible_matrices(self):
        matrices = [
            np.array([[1, 0], [0, 1]], dtype=np.float32),
            np.array([[1, 2], [3, 4]], dtype=np.float32),
            np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
        ]
        return matrices

    def create_matrix_inverse_net(self, input_shape, adjoint=False):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_tensor = tf.compat.v1.placeholder(np.float32, input_shape, 'input')
            output = tf.raw_ops.MatrixInverse(input=input_tensor, adjoint=adjoint)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 2]),
        dict(input_shape=[3, 3]),
        dict(input_shape=[4, 4]),
        dict(input_shape=[2, 4, 4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_inverse_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_matrix_inverse_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)