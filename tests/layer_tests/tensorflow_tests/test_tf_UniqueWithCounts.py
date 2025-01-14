# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestUnique(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        x_shape = inputs_info['x:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape)
        return inputs_data

    def create_unique_net(self, x_shape, data_type, out_idx):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(data_type, x_shape, 'x')
            unique = tf.raw_ops.UniqueWithCounts(x=x, out_idx=out_idx)
            tf.identity(unique[0], name='unique_elements')
            tf.identity(unique[1], name='unique_indices')
            tf.identity(unique[2], name='unique_counts')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[1], data_type=tf.float32, out_idx=tf.int32),
        dict(x_shape=[50], data_type=tf.float32, out_idx=tf.int32),
        dict(x_shape=[100], data_type=tf.float32, out_idx=tf.int64),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_unique_basic(self, params, ie_device, precision, ir_version, temp_dir,
                          use_legacy_frontend):
        if ie_device == 'GPU' and params['x_shape'] == [1]:
            pytest.skip("Input unique:UniqueWithCounts.out2 hasn't been found in primitive_ids map issue on GPU")
        if use_legacy_frontend:
            pytest.skip("Unique operation is not supported via legacy frontend.")
        self._test(*self.create_unique_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
