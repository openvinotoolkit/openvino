# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestExpandDims(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'input' in inputs_info, "Test error: inputs_info must contain `input`"
        x_shape = inputs_info['input']
        inputs_data = {}
        inputs_data['input'] = np.random.randint(-10, 10, x_shape).astype(np.float32)
        return inputs_data

    def create_expand_dims_net(self, input_shape, axis):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            axis = tf.constant(axis, dtype=tf.int32)
            tf.raw_ops.ExpandDims(input=input, axis=axis)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_basic = [
        dict(input_shape=[], axis=0),
        dict(input_shape=[2, 3], axis=1),
        dict(input_shape=[2, 3, 5], axis=-2),
    ]

    @pytest.mark.parametrize("params", test_basic)
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    def test_expand_dims_basic(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_expand_dims_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
