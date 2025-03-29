# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestBatchToSpace(CommonTFLayerTest):
    def create_batch_to_space_net(self, in_shape, crops_value, block_shape_value):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, in_shape, 'Input')
            crops = tf.constant(crops_value, dtype=tf.int32)
            block_shape = tf.constant(block_shape_value, dtype=tf.int32)
            tf.batch_to_space(input=x, block_shape=block_shape, crops=crops, name='Operation')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(in_shape=[4, 1, 1, 3], block_shape_value=[1], crops_value=[[0, 0]]),
        dict(in_shape=[12, 1, 1, 3], block_shape_value=[3, 1, 4], crops_value=[[1, 0], [0, 0], [1, 1]]),
        dict(in_shape=[72, 2, 1, 4, 2], block_shape_value=[3, 4, 2],
             crops_value=[[1, 2], [0, 0], [3, 0]]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_batch_to_space_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_batch_to_space_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_4D = [
        dict(in_shape=[4, 1, 1, 3], block_shape_value=[2, 2], crops_value=[[0, 0], [0, 0]]),
        dict(in_shape=[60, 100, 30, 30], block_shape_value=[3, 2], crops_value=[[1, 5], [4, 1]]),
        dict(in_shape=[4, 1, 1, 1], block_shape_value=[2, 1, 2], crops_value=[[0, 0], [0, 0], [0, 0]]),
        dict(in_shape=[36, 2, 2, 3], block_shape_value=[2, 3, 3], crops_value=[[1, 0], [0, 0], [2, 2]])
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_batch_to_space_4D(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_batch_to_space_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_5D = [
        dict(in_shape=[144, 2, 1, 4, 1], block_shape_value=[3, 4, 2, 2],
             crops_value=[[1, 2], [0, 0], [3, 0], [0, 0]]),
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_batch_to_space_5D(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_batch_to_space_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
