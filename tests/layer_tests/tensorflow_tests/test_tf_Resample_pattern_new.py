# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestResamplePattern(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(1, 256, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_resample_net(self, shape, factor, use_legacy_frontend):
        """
            The sub-graph in TF that could be expressed as a single Resample operation.
        """
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()
            tf_x_shape = np.array(tf_x_shape)[[0, 2, 3, 1]]

            input = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')

            transpose_1 = tf.transpose(a=input, perm=[1, 2, 3, 0])
            expand_dims = tf.expand_dims(transpose_1, 0)
            tile = tf.tile(expand_dims, [factor * factor, 1, 1, 1, 1])
            bts = tf.batch_to_space(tile, [factor, factor], [[0, 0], [0, 0]])
            strided_slice = bts[0, ...]
            tf.transpose(a=strided_slice, perm=[3, 0, 1, 2])

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [pytest.param(dict(shape=[1, 1, 100, 200], factor=2), marks=pytest.mark.precommit),
                 dict(shape=[1, 1, 200, 300], factor=3)]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_resample(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_resample_net(params['shape'], params['factor'], use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
