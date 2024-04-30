# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSpaceToDepth(CommonTFLayerTest):
    def create_space_to_depth_net(self, input_shape, block_size, data_format):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, input_shape, 'x')
            tf.raw_ops.SpaceToDepth(input=x, block_size=block_size, data_format=data_format)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 12, 16, 3], block_size=2, data_format='NHWC'),
        dict(input_shape=[3, 6, 15, 2], block_size=3, data_format='NHWC'),
        # On CPU TensorFlow does not support 'NCHW' for SpaceToDepth
        # dict(input_shape=[2, 3, 10, 10], block_size=5, data_format='NCHW'),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_space_to_depth_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_space_to_depth_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
