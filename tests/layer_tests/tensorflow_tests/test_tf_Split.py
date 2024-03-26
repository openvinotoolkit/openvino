# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSplit(CommonTFLayerTest):
    def create_split_net(self, value_shape, axis_value, num_split):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            axis = tf.constant(axis_value, dtype=tf.int32)
            value = tf.compat.v1.placeholder(tf.float32, value_shape, 'value')
            split = tf.raw_ops.Split(axis=axis, value=value, num_split=num_split)
            for output_ind in range(num_split):
                tf.identity(split[output_ind], name="split_" + str(output_ind))
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(value_shape=[6], axis_value=0, num_split=2),
        dict(value_shape=[2, 1, 6], axis_value=2, num_split=3),
        dict(value_shape=[4, 3, 2, 7], axis_value=-4, num_split=4),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_split_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        if ie_device == 'GPU' and params['value_shape'] == [4, 3, 2, 7]:
            pytest.skip("accuracy issue on GPU")
        self._test(*self.create_split_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
