# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestWhere(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'condition:0' in inputs_info, "Test error: inputs_info must contain `condition`"
        condition_shape = inputs_info['condition:0']
        inputs_data = {}
        inputs_data['condition:0'] = np.random.randint(-2, 2, condition_shape)
        return inputs_data

    def create_where_net(self, condition_shape, condition_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            condition = tf.compat.v1.placeholder(condition_type, condition_shape, 'condition')
            tf.raw_ops.Where(condition=condition, name='where')
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(condition_shape=[2], condition_type=tf.float32),
        dict(condition_shape=[3, 4], condition_type=tf.bool),
        dict(condition_shape=[2, 4, 5], condition_type=tf.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_where_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_where_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
