# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestBucketize(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info, "Test error: inputs_info must contain `input`"
        input_shape = inputs_info['input']
        input_type = self.input_type
        inputs_data = {}
        input_data = np.random.randint(-20, 20, input_shape).astype(input_type)
        inputs_data['input'] = input_data
        return inputs_data

    def create_bucketize_net(self, input_shape, input_type, boundaries_size):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            # generate boundaries list
            # use wider range for boundaries than input data in order to cover all bucket indices cases
            boundaries = np.sort(np.unique(np.random.randint(-200, 200, [boundaries_size]).astype(np.float32))).tolist()
            tf.raw_ops.Bucketize(input=input, boundaries=boundaries)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[5], input_type=np.int32, boundaries_size=1),
        dict(input_shape=[3, 4], input_type=np.float32, boundaries_size=0),
        dict(input_shape=[2, 3, 4], input_type=np.float32, boundaries_size=300),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(platform.machine() in ["aarch64", "arm64", "ARM64"],
                       reason='Ticket - 122716')
    def test_bucketize_basic(self, params, ie_device, precision, ir_version, temp_dir,
                             use_new_frontend):
        self._test(*self.create_bucketize_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
