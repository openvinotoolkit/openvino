# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import tensorflow as tf
import pytest
from common.tf_layer_test_class import CommonTFLayerTest

class TestApproximateEqual(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'tensor1:0' in inputs_info
        assert 'tensor2:0' in inputs_info
        tensor1_shape = inputs_info['tensor1:0']
        tensor2_shape = inputs_info['tensor2:0']
        inputs_data = {}
        inputs_data['tensor1:0'] = 4 * rng.random(tensor1_shape).astype(np.float32) - 2
        inputs_data['tensor2:0'] = 4 * rng.random(tensor2_shape).astype(np.float32) - 2
        return inputs_data

    def create_approximate_equal_net(self, input1_shape, input2_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tensor1 = tf.compat.v1.placeholder(tf.float32, input1_shape, 'tensor1')
            tensor2 = tf.compat.v1.placeholder(tf.float32, input2_shape, 'tensor2')
            approx_equal_op = tf.raw_ops.ApproximateEqual(x=tensor1, y=tensor2, tolerance=0.01)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input1_shape=[2, 3], input2_shape=[2, 3]),
        dict(input1_shape=[3, 4, 5], input2_shape=[3, 4, 5]),
        dict(input1_shape=[1, 2, 3, 4], input2_shape=[1, 2, 3, 4]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_approximate_equal_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                     use_legacy_frontend):
        if ie_device == 'GPU' and precision == 'FP16':
            pytest.skip("Accuracy mismatch on GPU")
        self._test(*self.create_approximate_equal_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
