# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTopKV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'input:0' in inputs_info, "Test error: inputs_info must contain `input`"
        x_shape = inputs_info['input:0']
        inputs_data = {}
        inputs_data['input:0'] = np.random.randint(-10, 10, x_shape)
        return inputs_data

    def create_topk_v2_net(self, input_shape, input_type, k, sorted, is_first_output, is_second_output):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            k = tf.constant(k, dtype=tf.int32, shape=[])
            topk = tf.raw_ops.TopKV2(input=input, k=k, sorted=sorted)
            if is_first_output:
                tf.identity(topk[0], name='topk_values')
            if is_second_output:
                tf.identity(topk[1], name='unique_indices')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, None

    test_basic = [
        dict(input_shape=[10], input_type=tf.float32, k=5, sorted=True, is_first_output=True, is_second_output=False),
        dict(input_shape=[2, 3, 10], input_type=tf.int32, k=10, sorted=True, is_first_output=True,
             is_second_output=False),
        dict(input_shape=[4, 12], input_type=tf.float32, k=10, sorted=True, is_first_output=True,
             is_second_output=True),
        # Expect stable mode implementation for sort_type=indices in OpenVINO. See 101503
        pytest.param(dict(input_shape=[5, 10], input_type=tf.int32, k=8, sorted=False, is_first_output=True,
                          is_second_output=True), marks=pytest.mark.xfail(reason="101503")),
    ]

    @pytest.mark.parametrize("params", test_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() in ('Linux', 'Darwin') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 126314, 122716')
    def test_topk_v2_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_topk_v2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
