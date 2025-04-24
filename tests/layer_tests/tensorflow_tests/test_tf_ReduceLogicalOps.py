# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest

# Testing Logical operations (Initial Implementation)
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/All
#                https://www.tensorflow.org/api_docs/python/tf/raw_ops/Any

class TestLogicalOps(CommonTFLayerTest):
    # Overload inputs generation to fill dummy input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(0, 2, inputs_dict[input]).astype(bool)
        return inputs_dict

    # input_shape - should be an array
    # axis - array which points on axis for the operation
    # op_type - type of tested operation
    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    def create_logical_ops_placeholder_const_net(self, input_shape, axis, op_type, ir_version, use_legacy_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->op_type   =>       Placeholder->ReduceLogicalAnd/Or
                         /                               /
            Const-------/                   Const-------/

        """
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            op_type_to_tf = {
                'All': tf.raw_ops.All,
                'Any': tf.raw_ops.Any,
            }
            tf_input = tf.compat.v1.placeholder(tf.bool, input_shape, 'Input')
            tf_axis = tf.constant(axis)

            op_type_to_tf[op_type](input = tf_input, axis = tf_axis)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        pytest.param(
            dict(input_shape=[2, 3], axis=[1]),     #Simple test
            marks=pytest.mark.precommit),
        dict(input_shape=[2, 3, 3, 4], axis=[2]),   #Simple test with possible nchw/nhwc
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("op_type", ['All', 'Any'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_logical_ops_placeholder_const(self, params, op_type, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_logical_ops_placeholder_const_net(**params, op_type=op_type, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
