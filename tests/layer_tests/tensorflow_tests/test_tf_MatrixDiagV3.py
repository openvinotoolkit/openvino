# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestMatrixDiagV3(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        logger.info(inputs_info)
        assert 'diagonal:0' in inputs_info
        diagonal_shape = inputs_info['diagonal:0']
        inputs_data = {}
        inputs_data['diagonal:0'] = np.random.randint(-50, 50, diagonal_shape).astype(self.diagonal_type)
        logger.info(inputs_data)
        return inputs_data

    def create_matrix_diag_v3_net(self, diagonal_type, diagonal_shape, k, num_rows, num_cols, padding_value, align="RIGHT_LEFT"):
        self.diagonal_type = diagonal_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            diagonal = tf.compat.v1.placeholder(diagonal_type, diagonal_shape, 'diagonal')
            params = {
                'diagonal': diagonal,
                'k': k,
                'num_rows': num_rows,
                'num_cols': num_cols,
                'padding_value': padding_value,
                # 'align': align,
            }
            logger.info(k)
            k = tf.constant(k)
            tf.raw_ops.MatrixDiagV3(diagonal=diagonal, k=k, num_rows=num_rows, num_cols=num_cols, padding_value=padding_value)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        
        return tf_net, None

    @pytest.mark.parametrize("test_params", [
    {
        "diagonal_type": np.int64,
        "diagonal_shape": [1, 3],
        "k": 3,
        "num_cols": -1,
        "num_rows": -1,
        "padding_value": -1,
        # "align": "RIGHT_LEFT" 
    }
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_diag_v3_basic(self, test_params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("GPU support issue in local development")
        self._test(*self.create_matrix_diag_v3_net(**test_params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
