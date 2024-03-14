# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


# Testing operation ArgMin, ArgMax (Initial Implementation)
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMin
#                https://www.tensorflow.org/api_docs/python/tf/raw_ops/ArgMax

OPS = {
        'tf.raw_ops.ArgMax': tf.raw_ops.ArgMax,
        'tf.raw_ops.ArgMin': tf.raw_ops.ArgMin
}

class TestArgMinMax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info
        input_shape = inputs_info['input:0']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['input:0'] = rng.integers(-8, 8, input_shape).astype(self.input_type)
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
        [[20], 0],
        [[20, 30], 1],
        [[2, 30, 3, 4], 2],
    ]

    @pytest.mark.parametrize("input_shape, dimension", test_data)
    @pytest.mark.parametrize("input_type", [np.float32, np.int32])
    @pytest.mark.parametrize("output_type", [tf.int32, tf.int64])
    @pytest.mark.parametrize("op_type", ['tf.raw_ops.ArgMax', 'tf.raw_ops.ArgMin'])
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ['arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'],
                       reason='Ticket - 126314, 132699')
    def test_argmin_max_net(self, input_shape, dimension, input_type, output_type, op_type, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        params = dict(input_shape=input_shape, dimension=dimension)
        self._test(*self.create_argmin_max_net(**params, input_type=input_type,
                                               output_type=output_type, op_type=OPS[op_type]),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
