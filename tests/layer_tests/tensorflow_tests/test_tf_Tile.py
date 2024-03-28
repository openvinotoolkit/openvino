# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTile(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        assert 'multiples:0' in inputs_info
        input_shape = inputs_info['input:0']
        multiples_shape = inputs_info['multiples:0']
        inputs_data = {}
        inputs_data['input:0'] = np.random.randint(-50, 50, input_shape).astype(np.float32)
        inputs_data['multiples:0'] = np.random.randint(1, 4, multiples_shape).astype(np.int32)

        return inputs_data

    def create_tile_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            multiples = tf.compat.v1.placeholder(tf.int32, [len(input_shape)], 'multiples')
            tf.raw_ops.Tile(input=input, multiples=multiples)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 4]),
        dict(input_shape=[3, 1, 2]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_tile_basic(self, params, ie_device, precision, ir_version, temp_dir,
                        use_legacy_frontend):
        self._test(*self.create_tile_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
