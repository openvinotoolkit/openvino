# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestOnesLike(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x' in inputs_info
        x_shape = inputs_info['x']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['x'] = rng.integers(-10, 10, x_shape).astype(self.x_type)
        return inputs_data

    def create_ones_like_net(self, x_shape, x_type):
        self.x_type = x_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.dtypes.as_dtype(x_type), x_shape, 'x')
            tf.raw_ops.OnesLike(x=x)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[], x_type=np.float32),
        dict(x_shape=[2], x_type=np.int32),
        dict(x_shape=[2, 3, 4], x_type=np.float32),
        dict(x_shape=[1, 4, 3, 1], x_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_ones_like(self, params, ie_device, precision, ir_version, temp_dir,
                       use_new_frontend):
        self._test(*self.create_ones_like_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
