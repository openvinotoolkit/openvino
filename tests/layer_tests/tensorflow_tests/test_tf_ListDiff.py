# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestListDiff(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info, "Test error: inputs_info must contain `x`"
        assert 'y:0' in inputs_info, "Test error: inputs_info must contain `y`"
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        inputs_data['x:0'] = np.random.randint(-5, 5, x_shape)
        inputs_data['y:0'] = np.random.randint(-5, 5, y_shape)
        return inputs_data

    def create_list_diff_net(self, x_shape, y_shape, out_idx):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.int32, x_shape, 'x')
            y = tf.compat.v1.placeholder(tf.int32, y_shape, 'y')
            listdiff = tf.raw_ops.ListDiff(x=x, y=y, out_idx=out_idx)
            tf.identity(listdiff[0], name='out')
            tf.identity(listdiff[1], name='idx')
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[7], y_shape=[2], out_idx=tf.int32),
        dict(x_shape=[4], y_shape=[8], out_idx=tf.int64),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_list_diff_basic(self, params, ie_device, precision, ir_version, temp_dir,
                             use_legacy_frontend):
        self._test(*self.create_list_diff_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
