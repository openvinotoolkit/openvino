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
        inputs_data['input:0'] = np.random.rand(*input_shape).astype(np.float32)  # Example random input

        return inputs_data

    def create_matrix_inverse_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input_tensor = tf.compat.v1.placeholder(np.float32, input_shape, 'input')
            output = tf.raw_ops.MatrixInverse(input=input_tensor, adjoint=False)  # Adjust 'adjoint' as needed
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 2]),
        dict(input_shape=[3, 3]),
        dict(input_shape=[4, 4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_inverse_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_matrix_inverse_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)