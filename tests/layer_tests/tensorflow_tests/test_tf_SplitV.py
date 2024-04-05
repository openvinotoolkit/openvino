# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import os

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSplitV(CommonTFLayerTest):
    def create_splitv_net(self, value_shape, size_splits_values, axis_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            axis = tf.constant(axis_value, dtype=tf.int32)
            size_splits = tf.constant(size_splits_values, dtype=tf.int32)
            value = tf.compat.v1.placeholder(tf.float32, value_shape, 'value')
            num_split = len(size_splits_values)
            splitv = tf.raw_ops.SplitV(value=value, size_splits=size_splits, axis=axis, num_split=num_split)
            for output_ind in range(num_split):
                if size_splits_values[output_ind] != 0:
                    tf.identity(splitv[output_ind], name="split_" + str(output_ind))
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(value_shape=[3], size_splits_values=[0, 2, 0, 1], axis_value=0),
        dict(value_shape=[2, 3, 9], size_splits_values=[1, 2, 3, -1, 1], axis_value=2),
        dict(value_shape=[3, 9, 5, 4], size_splits_values=[1, 2, 0, -1, 2, 0], axis_value=-3),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") == 'true', reason="Ticket - 113359")
    def test_split_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        if ie_device == 'GPU' and params['value_shape'] == [3]:
            pytest.skip("accuracy issue on GPU")
        self._test(*self.create_splitv_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
