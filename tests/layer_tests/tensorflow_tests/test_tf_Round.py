# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestRound(CommonTFLayerTest):
    def create_tf_round_net(self, input_shape, input_type):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Round(x=input)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize("input_shape", [[6], [2, 5, 3], [10, 5, 1, 5]])
    @pytest.mark.parametrize("input_type", [tf.float32, tf.int32, tf.int64, tf.float64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_round_basic(self, input_shape, input_type, ie_device, precision,
                        ir_version, temp_dir, use_legacy_frontend):
        if ie_device == 'GPU' and input_type in [tf.int32, tf.int64]:
            pytest.skip("GPU error: Requested activation is not supported for integer type")
        self._test(*self.create_tf_round_net(input_shape, input_type),
                ie_device, precision, ir_version, temp_dir=temp_dir,
                use_legacy_frontend=use_legacy_frontend)
