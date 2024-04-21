# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestMatrixDiag(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'diagonal:0' in inputs_info
        diagonal_shape = inputs_info['diagonal:0']
        inputs_data = {}
        inputs_data['diagonal:0'] = np.random.randint(-50, 50, diagonal_shape).astype(self.diagonal_type)

        return inputs_data

    def create_matrix_diag_net(self, diagonal_shape, diagonal_type):
        self.diagonal_type = diagonal_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            diagonal = tf.compat.v1.placeholder(diagonal_type, diagonal_shape, 'diagonal')
            tf.raw_ops.MatrixDiag(diagonal=diagonal)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(diagonal_shape=[5], diagonal_type=np.float32),
        dict(diagonal_shape=[3, 1], diagonal_type=np.int32),
        dict(diagonal_shape=[2, 1, 4, 3], diagonal_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_diag_basic(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_matrix_diag_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
