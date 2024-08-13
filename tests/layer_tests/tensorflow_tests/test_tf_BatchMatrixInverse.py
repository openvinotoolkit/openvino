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
        x = np.random.rand(*x_shape).astype(np.float32)
        for idx in np.ndindex(x_shape[:-2]):
            x[idx] += np.eye(x_shape[-1]) * x_shape[-1]
        inputs_data['x:0'] = x
        return inputs_data

    def create_batch_matrix_inverse_net(self, input_shape, data_type):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(data_type, input_shape, 'x')
            y = tf.linalg.inv(x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
        dict(input_shape=(2, 3, 3), data_type=tf.float32),
        dict(input_shape=(2, 4, 4), data_type=tf.float32),
        dict(input_shape=(5, 2, 2), data_type=tf.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_batch_matrix_inverse_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_batch_matrix_inverse_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
