# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation ArgMin, ArgMax (Initial Implementation)
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMin
#                https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMax

class TestArgMinMax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input' in inputs_info
        input_shape = inputs_info['input']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['input'] = rng.integers(-8, 8, input_shape).astype(self.input_type)
        return inputs_data

    def create_argmin_max_net(self, input_shape, dimension, input_type, output_type, op_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf_dimension = tf.constant(dimension)

            op_type(input=tf_input, dimension=tf_dimension, output_type=output_type)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[20], dimension=0),
        dict(input_shape=[20, 30], dimension=1),
        dict(input_shape=[2, 30, 3, 4], dimension=2),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    @pytest.mark.parametrize("output_type", [tf.int32, tf.int64])
    @pytest.mark.parametrize("op_type", [tf.raw_ops.ArgMax, tf.raw_ops.ArgMin])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_argmin_max_net(self, params, input_type, output_type, op_type, ie_device, precision, ir_version, temp_dir,
                            use_new_frontend, use_old_api):
        self._test(*self.create_argmin_max_net(**params, input_type=input_type,
                                               output_type=output_type, op_type=op_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
