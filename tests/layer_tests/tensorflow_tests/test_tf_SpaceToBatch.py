# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestSpaceToBatch(CommonTFLayerTest):
    def create_space_to_batch_net(self, in_shape, pads_value, block_shape_value):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, in_shape, 'Input')
            pads = tf.constant(pads_value, dtype=tf.int32)
            block_shape = tf.constant(block_shape_value, dtype=tf.int32)
            tf.space_to_batch_nd(x, block_shape, pads, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(in_shape=[4, 1, 1, 3], block_shape_value=[1], pads_value=[[0, 0]]),
        dict(in_shape=[2, 3, 6, 5], block_shape_value=[2, 3, 3], pads_value=[[1, 0], [0, 0], [2, 2]]),
        dict(in_shape=[1, 2, 9, 1], block_shape_value=[4, 3], pads_value=[[1, 1], [2, 4]]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_space_to_batch_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_space_to_batch_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_4D = [
        dict(in_shape=[1, 2, 2, 3], block_shape_value=[2, 2], pads_value=[[0, 0], [0, 0]]),
        dict(in_shape=[1, 2, 1, 4], block_shape_value=[3, 2, 2], pads_value=[[1, 0], [0, 1], [1, 1]])
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_space_to_batch_4D(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_space_to_batch_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_5D = [
        dict(in_shape=[3, 3, 4, 5, 2], block_shape_value=[3, 4, 2],
             pads_value=[[1, 2], [0, 0], [3, 0]]),
        dict(in_shape=[3, 3, 4, 5, 2], block_shape_value=[3, 4, 2, 2],
             pads_value=[[1, 2], [0, 0], [3, 0], [0, 0]]),
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_space_to_batch_5D(self, params, ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        self._test(*self.create_space_to_batch_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
