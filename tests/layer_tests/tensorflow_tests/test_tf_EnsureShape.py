# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestEnsureShape(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'tensor:0' in inputs_info
        tensor_shape = inputs_info['tensor:0']
        inputs_data = {}
        inputs_data['tensor:0'] = np.random.randint(-10, 10, tensor_shape).astype(self.input_type)
        return inputs_data

    def create_ensure_shape_net(self, input_shape, input_type, target_shape):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tensor = tf.compat.v1.placeholder(input_type, input_shape, 'tensor')
            shape = tf.constant(target_shape, dtype=tf.int32)
            reshape = tf.raw_ops.Reshape(tensor=tensor, shape=shape)
            tf.raw_ops.EnsureShape(input=reshape, shape=target_shape)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 6], input_type=np.float32, target_shape=[2, 3, 2]),
        dict(input_shape=[1], input_type=np.float32, target_shape=[]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_ensure_shape_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("timeout issue on GPU")
        self._test(*self.create_ensure_shape_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
