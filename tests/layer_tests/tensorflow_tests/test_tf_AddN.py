# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest

import logging

# Testing operation AddN
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/AddN

class TestAddN(CommonTFLayerTest):
    # input_shapes - should be an array, could be a single shape, or array of n-dimentional shapes
    # ir_version - common parameter
    # use_new_frontend - common parameter
    def create_addn_placeholder_const_net(self, input_shapes, ir_version, use_new_frontend):
        """
            Tensorflow net                  IR net

            Placeholder_1->AddN    =>       Placeholder_1->AddN
            ...           /                 ...           /
            Placeholder_N/                  Placeholder_N/

        """

        if len(input_shapes) == 0:
            raise RuntimeError("Input list couldn't be empty")

        if len(input_shapes) == 1 and not use_new_frontend:
            pytest.xfail(reason="96687")
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_inputs = []

            for idx, input_shape in enumerate(input_shapes):
                tf_inputs.append(tf.compat.v1.placeholder(tf.float32, input_shape, f"Input_{idx}"))

            tf.raw_ops.AddN(inputs = tf_inputs)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        dict(input_shapes=[[4]]),                             # Tests sum of scalar values in a single shape
        pytest.param(
            dict(input_shapes=[[4, 3], [4, 3]]),              # Tests sum of shapes
            marks=pytest.mark.precommit_tf_fe),
        dict(input_shapes=[[3, 4, 5], [3, 4, 5], [3, 4, 5]]), # Tests sum of shapes which may trigger nchw/nhcw transformation
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_addn_placeholder_const(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend):
        self._test(*self.create_addn_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
