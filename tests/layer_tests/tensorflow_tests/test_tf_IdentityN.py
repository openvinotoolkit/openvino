# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestIdentityN(CommonTFLayerTest):
    def create_identityn_net(self, value_shape, size_splits_values, axis_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            axis = tf.constant(axis_value, dtype=tf.int32)
            size_splits = tf.constant(size_splits_values, dtype=tf.int32)
            value = tf.compat.v1.placeholder(tf.float32, value_shape, 'value')
            num_split = len(size_splits_values)
            # we have to do a trick using Split operation to avoid fusing output tensors into input tensors
            # so that Parameter node will contain both input and output tensor names
            # 97192: this is a limitation of layer test infrastructure
            split = tf.raw_ops.SplitV(value=value, size_splits=size_splits, axis=axis, num_split=num_split)
            split_outputs = []
            for output_ind in range(num_split):
                split_outputs.append(split[output_ind])
            identity_n = tf.raw_ops.IdentityN(input=split_outputs)
            for output_ind in range(num_split):
                tf.identity(identity_n[output_ind], name="Identity_" + str(output_ind))

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(value_shape=[2, 3, 9], size_splits_values=[1, 2, 3, -1, 1], axis_value=2),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_split_basic(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(*self.create_identityn_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
