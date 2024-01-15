# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestFloorDiv(CommonTFLayerTest):
    def create_add_placeholder_const_net(self, x_shape, dtype, ir_version, use_new_frontend):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(dtype, x_shape, 'Input')
            constant_value = np.array(-10).astype(dtype)
            y = tf.constant(constant_value)
            x = tf.raw_ops.Abs(x=x)
            res = tf.raw_ops.FloorDiv(x=x, y=y)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # TODO: implement tests for 2 Consts + Add

    test_data_1D = [
        dict(x_shape=[], dtype=np.int32),
        dict(x_shape=[2], dtype=np.int64),
        dict(x_shape=[2, 4, 5], dtype=np.int32),
        dict(x_shape=[], dtype=np.float32),
        dict(x_shape=[2], dtype=np.float64),
        dict(x_shape=[2, 4, 5], dtype=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    def test_add_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
