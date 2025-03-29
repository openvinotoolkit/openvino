# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestClipByValue(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 't:0' in inputs_info, "Test error: inputs_info must contain `t`"
        t_shape = inputs_info['t:0']
        clip_value_min_shape = inputs_info['clip_value_min:0']
        clip_value_max_shape = inputs_info['clip_value_max:0']
        inputs_data = {}
        inputs_data['t:0'] = np.random.randint(-10, 10, t_shape).astype(np.float32)
        inputs_data['clip_value_min:0'] = np.random.randint(-10, 10, clip_value_min_shape).astype(np.float32)
        inputs_data['clip_value_max:0'] = np.random.randint(-10, 10, clip_value_max_shape).astype(np.float32)
        return inputs_data

    def create_clip_by_value_net(self, t_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            t = tf.compat.v1.placeholder(tf.float32, t_shape, 't')
            clip_value_min = tf.compat.v1.placeholder(tf.float32, [], 'clip_value_min')
            clip_value_max = tf.compat.v1.placeholder(tf.float32, [], 'clip_value_max')
            tf.raw_ops.ClipByValue(t=t, clip_value_min=clip_value_min, clip_value_max=clip_value_max)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(t_shape=[]),
        dict(t_shape=[2, 3]),
        dict(t_shape=[4, 1, 3]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_clip_by_value_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        self._test(
            *self.create_clip_by_value_net(**params), ie_device,
            precision, temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
            **params)
