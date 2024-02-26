# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestMatrixBandPart(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        inputs_data['input:0'] = np.random.randint(-50, 50, input_shape).astype(np.float32)
        return inputs_data

    def create_matrix_band_part_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_tensor = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            tf.raw_ops.MatrixBandPart(input=input_tensor, num_lower=-1, num_upper=0)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        return tf_net, None

    test_data_basic = [
        dict(input_shape=[5, 5]),
        dict(input_shape=[3, 4, 4]),
        dict(input_shape=[2, 3, 5, 5]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_matrix_band_part_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_matrix_band_part_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
