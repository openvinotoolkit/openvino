# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestRank(CommonTFLayerTest):
    def create_rank_net(self, input_shape, input_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Rank(input=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[], input_type=tf.float32),
        dict(input_shape=[1], input_type=tf.float32),
        dict(input_shape=[2, 6], input_type=tf.int32),
        dict(input_shape=[3, 4, 5, 6], input_type=tf.float32)
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_rank_basic(self, params, ie_device, precision, ir_version, temp_dir,
                        use_new_frontend):
        self._test(*self.create_rank_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
