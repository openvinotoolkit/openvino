# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest

# Testing BroadcastTo operation
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/BroadcastTo

class TestBroadcastTo(CommonTFLayerTest):
    # input_shape - shape a tensor for a broadcast
    # broadcast_shape - shape of output
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_broadcastto_placeholder_const_net(self, input_shape, broadcast_shape, ir_version, use_legacy_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->BroadcastTo   =>   Placeholder->Broadcast

        """
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf.int32, input_shape, 'Input')

            tf.raw_ops.BroadcastTo(input = tf_input, shape = broadcast_shape)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[3], broadcast_shape=[2, 3]),
        dict(input_shape=[3, 4, 5], broadcast_shape = [4, 3, 3, 4, 5]),
        dict(input_shape=[1, 2, 3, 5], broadcast_shape = [4, 1, 2, 3, 5])
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_broadcastto_placeholder_const(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_broadcastto_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)