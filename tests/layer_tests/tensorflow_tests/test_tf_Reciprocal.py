# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestReciprocal(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info
        x_shape = inputs_info['x']
        inputs_data = {}
        inputs_data['x'] = np.random.randint(1, 30, x_shape).astype(np.float32)

        return inputs_data

    def create_reciprocal_net(self, x_shape, x_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            tf.raw_ops.Reciprocal(x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[], x_type=np.float32),
        dict(x_shape=[2, 3], x_type=np.float32),
        dict(x_shape=[4, 1, 3], x_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_reciprocal_basic(self, params, ie_device, precision, ir_version, temp_dir,
                              use_new_frontend, use_old_api):
        self._test(*self.create_reciprocal_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
