# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest

# Testing Cumsum operation
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Cumsum

class TestCumsumOps(CommonTFLayerTest):
    # input_shape - should be an array
    # axis - array which points on axis for the operation
    # exclusive - enables exclusive Cumsum
    # reverse - enables reverse order of Cumsum
    # ir_version - common parameter
    # use_new_frontend - common parameter
    def create_cumsum_ops_placeholder_const_net(self, input_shape, axis, exclusive, reverse, ir_version, use_new_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->Cumsum   =>       Placeholder->Cumsum

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(tf.float32, input_shape, 'Input')
            tf_axis = tf.constant(axis)

            tf.raw_ops.Cumsum(x = tf_input, axis = tf_axis, exclusive = exclusive, reverse = reverse)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        pytest.param(
            dict(input_shape=[2, 3], axis=1),
            marks=pytest.mark.precommit_tf_fe),
        dict(input_shape=[2, 3, 3, 4], axis=2),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("exclusive", [False, True])
    @pytest.mark.parametrize("reverse", [False, True])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_cumsum_ops_placeholder_const(self, params, exclusive, reverse, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_cumsum_ops_placeholder_const_net(**params, exclusive=exclusive, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, reverse=reverse),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)