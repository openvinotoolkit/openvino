# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestShapeN(CommonTFLayerTest):
    def create_shape_n_net(self, input_shapes, out_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            inputs = []
            for ind, input_shape in enumerate(input_shapes):
                inputs.append(tf.compat.v1.placeholder(tf.float32, input_shape, 'input_{}'.format(ind)))

            shapen = tf.raw_ops.ShapeN(input=inputs, out_type=out_type)
            tf.raw_ops.ConcatV2(values=shapen, axis=0)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shapes=[[2, 3], [1]], out_type=tf.int32),
        dict(input_shapes=[[3], [3, 2, 1], [], [4, 3, 1, 1]], out_type=tf.int64),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_shape_n_basic(self, params, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        self._test(*self.create_shape_n_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
