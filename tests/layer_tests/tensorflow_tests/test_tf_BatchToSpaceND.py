# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestBatchToSpaceND(CommonTFLayerTest):
    def create_batch_to_space_nd_net(self, input_shape, block_shape, crops):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, input_shape, 'x')
            tf.raw_ops.BatchToSpaceND(input=x, block_shape=block_shape, crops=crops)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[4, 1, 1, 3], block_shape=[1], crops=[[0, 0]]),
        dict(input_shape=[6, 14, 13, 3], block_shape=[1, 2], crops=[[1, 5], [4, 1]]),
        dict(input_shape=[72, 2, 1, 4, 2], block_shape=[3, 4, 2],
             crops=[[1, 2], [0, 0], [3, 0]]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_batch_to_space_nd_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                     use_legacy_frontend):
        self._test(*self.create_batch_to_space_nd_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
