# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestOneHot(CommonTFLayerTest):
    @staticmethod
    def create_one_hot_net(shape, depth, on_value, off_value, axis):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            indices = tf.compat.v1.placeholder(tf.int32, shape, name='input_indices')
            tf.raw_ops.OneHot(indices=indices,
                              depth=depth,
                              on_value=on_value,
                              off_value=off_value,
                              axis=axis,
                              name='Operation')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(shape=[2], depth=3, on_value=1.0, off_value=-1.0, axis=None),
        dict(shape=[2, 3], depth=4, on_value=5.0, off_value=10.0, axis=1),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_one_hot_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_one_hot_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_complex = [
        dict(shape=[3, 4], depth=1, on_value=1.0, off_value=-5.0, axis=-2),
        dict(shape=[3, 4, 2, 1], depth=4, on_value=3.0, off_value=5.0, axis=2),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.nightly
    def test_one_hot_complex(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_one_hot_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
