# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSize(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        input_type = self.input_type
        inputs_data = {}
        inputs_data['input'] = np.random.randint(-50, 50, input_shape).astype(input_type)

        return inputs_data

    def create_size_net(self, input_shape, input_type, out_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Size(input=input, out_type=out_type)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[], input_type=np.float32, out_type=tf.int32),
        dict(input_shape=[2, 3], input_type=np.int32, out_type=None),
        dict(input_shape=[4, 1, 3], input_type=np.float32, out_type=tf.int64),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_size_basic(self, params, ie_device, precision, ir_version, temp_dir,
                        use_new_frontend):
        self._test(*self.create_size_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
