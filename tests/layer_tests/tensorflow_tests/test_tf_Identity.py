# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


OPS = {
    'tf.raw_ops.Identity': tf.raw_ops.Identity,
    'tf.raw_ops.PreventGradient': tf.raw_ops.PreventGradient,
    'tf.raw_ops.Snapshot': tf.raw_ops.Snapshot,
    'tf.raw_ops.StopGradient': tf.raw_ops.StopGradient,
}

class TestIdentity(CommonTFLayerTest):
    def create_identity_net(self, input_shape, identity_op):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            relu = tf.raw_ops.Relu(features=input)
            identity_op(input=relu, name="identity")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        [[2], 'tf.raw_ops.Identity'],
        [[2, 3], 'tf.raw_ops.PreventGradient'],
        [[], 'tf.raw_ops.Snapshot'],
        [[1, 2, 3], 'tf.raw_ops.StopGradient']
    ]

    @pytest.mark.parametrize("input_shape, identity_op", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_identity_basic(self, input_shape, identity_op, ie_device, precision, ir_version, temp_dir,
                            use_legacy_frontend):
        params = dict(input_shape=input_shape, identity_op=OPS[identity_op])
        self._test(*self.create_identity_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
