# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()


class TestMatrixBandPart(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input:0']
        inputs_data = {}
        inputs_data['input:0'] = rng.integers(-8, 8, input_shape).astype(self.input_type)
        return inputs_data

    def create_matrix_band_part_net(self, input_shape, input_type, num_lower, num_upper):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input_tensor = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.MatrixBandPart(input=input_tensor, num_lower=num_lower, num_upper=num_upper)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def
        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[5, 5], [3, 4, 4], [1, 2, 5, 5], [3, 5, 4]])
    @pytest.mark.parametrize('input_type', [np.float32, np.int32])
    @pytest.mark.parametrize('num_lower', [-4, -1, 0, 1, 4])
    @pytest.mark.parametrize('num_upper', [-4, -1, 0, 1, 4])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_matrix_band_part_basic(self, input_shape, input_type, num_lower, num_upper,
                                    ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        self._test(*self.create_matrix_band_part_net(input_shape, input_type, num_lower, num_upper),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
