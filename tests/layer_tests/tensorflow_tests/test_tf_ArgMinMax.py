# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest

# Testing operation ArgMin, ArgMax (Initial Implementation)
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMin
#                https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMax

class TestArgMinMax(CommonTFLayerTest):
    # input_shape - should be an array
    # dimension - dimension to be used, for vector should be 0
    # op_type - type of testing operation
    # ir_version - common parameter
    # use_new_frontend - common parameter
    def create_argminmax_placeholder_const_net(self, input_shape, dimension, op_type, ir_version, use_new_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->op_type    =>       Placeholder->TopK->Squeeze
                         /                               /     /
            Const-------/                   Const-------/-----/

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            op_type_to_tf = {
                'ArgMax': tf.raw_ops.ArgMax,
                'ArgMin': tf.raw_ops.ArgMin,
            }
            tf_input_shape = input_shape.copy()
            tf_input = tf.compat.v1.placeholder(tf.float32, tf_input_shape, 'Input')
            tf_dimension = tf.constant(dimension)

            op_type_to_tf[op_type](input = tf_input, dimension = tf_dimension)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shape=[5], dimension=0),            #Simple test of vector
        pytest.param(
            dict(input_shape=[2, 3], dimension=1),     #Simple test
            marks=pytest.mark.precommit_tf_fe),
        dict(input_shape=[2, 3, 3, 4], dimension=2),   #Simple test with possible nchw/nhcw
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("op_type", ['ArgMin', 'ArgMax'])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_argminmax_placeholder_const(self, params, op_type, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_argminmax_placeholder_const_net(**params, op_type=op_type,
                                                                ir_version=ir_version,
                                                                use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
