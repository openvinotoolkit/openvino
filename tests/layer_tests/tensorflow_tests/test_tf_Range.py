# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestRange(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}
        if self.negative_delta:
            inputs_data['start:0'] = np.random.randint(1, 10, []).astype(self.input_type)
            inputs_data['limit:0'] = np.random.randint(-10, 0, []).astype(self.input_type)
            inputs_data['delta:0'] = np.random.randint(-5, -1, []).astype(self.input_type)
        else:
            inputs_data['start:0'] = np.random.randint(1, 10, []).astype(self.input_type)
            inputs_data['limit:0'] = np.random.randint(10, 30, []).astype(self.input_type)
            inputs_data['delta:0'] = np.random.randint(1, 5, []).astype(self.input_type)

        return inputs_data

    def create_range_net(self, input_type, negative_delta):
        self.input_type = input_type
        self.negative_delta = negative_delta
        types_map = {
            np.float32: tf.float32,
            np.int32: tf.int32
        }
        assert input_type in types_map, "Incorrect test case"
        tf_type = types_map[input_type]
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            start = tf.compat.v1.placeholder(tf_type, [], 'start')
            limit = tf.compat.v1.placeholder(tf_type, [], 'limit')
            delta = tf.compat.v1.placeholder(tf_type, [], 'delta')
            tf.raw_ops.Range(start=start, limit=limit, delta=delta)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_type=np.float32, negative_delta=False),
        dict(input_type=np.int32, negative_delta=True),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_range_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_range_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
