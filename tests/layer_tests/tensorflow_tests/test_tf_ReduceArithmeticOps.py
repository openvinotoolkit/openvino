# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestReduceArithmeticOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info, "Test error: inputs_info must contain `input`"
        x_shape = inputs_info['input']
        inputs_data = {}
        inputs_data['input'] = np.random.randint(-10, 10, x_shape).astype(np.float32)
        return inputs_data

    def create_reduce_net(self, shape, axis, operation, keep_dims, ir_version, use_new_frontend):
        import tensorflow as tf
        ops_mapping = {
            "Max": tf.raw_ops.Max,
            "Mean": tf.raw_ops.Mean,
            "Min": tf.raw_ops.Min,
            "Prod": tf.raw_ops.Prod,
            "Sum": tf.raw_ops.Sum,
            "EuclideanNorm": tf.raw_ops.EuclideanNorm
        }
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, shape, 'input')
            ops_mapping[operation](input=input, axis=axis, keep_dims=keep_dims, name="reduce")
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data = [
        dict(shape=[5], axis=0),
        dict(shape=[2, 3, 5], axis=1),
        dict(shape=[3, 1, 2, 4], axis=-2),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("operation", ["EuclideanNorm", "Max", "Mean", "Min", "Prod", "Sum"])
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    @pytest.mark.precommit_tf_fe
    def test_reduce(self, params, operation, keep_dims, ie_device, precision, ir_version, temp_dir,
                    use_new_frontend):
        self._test(*self.create_reduce_net(**params, operation=operation, keep_dims=keep_dims, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
