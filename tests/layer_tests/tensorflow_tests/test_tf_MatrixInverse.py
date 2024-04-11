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

        self.invertible_matrices = self._generate_invertible_matrices(input_shape)
        inputs_data['input:0'] = np.stack(self.invertible_matrices, axis=0)

        return inputs_data
    
    def _generate_invertible_matrices(self, input_shape):
        matrices = [
            np.arange(4).reshape(2, 2),
            np.arange(9).reshape(3, 3),
            np.arange(16).reshape(4, 4),
            [np.arange(16).reshape(4, 4), np.arange(16, 32).reshape(4, 4)],
        ]
        if len(input_shape) == 2:
            if len(input_shape[0]) == 2:
                matrix = matrices[0]
            elif len(input_shape[0]) == 3:
                matrix = matrices[1]
            elif len(input_shape[0]) == 4:
                matrix = matrices[2]
        else:
            matrix = matrix[3]
        return matrix

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
    @pytest.mark.parametrize("adjoint", [True, False])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_inverse_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_matrix_inverse_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)