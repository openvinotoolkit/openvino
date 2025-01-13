# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestBroadcastArgs(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 's0:0' in inputs_info, "Test error: inputs_info must contain `s0`"
        assert 's1:0' in inputs_info, "Test error: inputs_info must contain `s1`"
        s0_shape = inputs_info['s0:0']
        s1_shape = inputs_info['s1:0']
        inputs_data = {}
        inputs_data['s0:0'] = np.random.randint(1, 6, s0_shape)
        inputs_data['s1:0'] = np.random.randint(1, 6, s1_shape)
        # compute mask where we need to change dimension size in s1
        # so that s1 will be broadcastable to s0
        non_one_mask = inputs_data['s0:0'] != 1
        diff_size = len(inputs_data['s1:0']) - len(inputs_data['s0:0'])
        if diff_size > 0:
            # pad False elements to non_one_mask to the begin
            pad = np.full([diff_size], False, dtype=bool)
            non_one_mask = np.concatenate((pad, non_one_mask), axis=0)
        else:
            # cut extra mask elements
            diff_size = abs(diff_size)
            non_one_mask = non_one_mask[diff_size:]
        update_inds = np.argwhere(non_one_mask)
        inputs_data['s1:0'][update_inds] = 1

        return inputs_data

    def create_broadcast_args_net(self, s0_shape, s1_shape, input_type):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            s0 = tf.compat.v1.placeholder(input_type, s0_shape, 's0')
            s1 = tf.compat.v1.placeholder(input_type, s1_shape, 's1')
            tf.raw_ops.BroadcastArgs(s0=s0, s1=s1)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
        dict(s0_shape=[6], s1_shape=[6], input_type=tf.int32),
        dict(s0_shape=[2], s1_shape=[5], input_type=tf.int64),
        dict(s0_shape=[7], s1_shape=[1], input_type=tf.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_broadcast_args_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_broadcast_args_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
