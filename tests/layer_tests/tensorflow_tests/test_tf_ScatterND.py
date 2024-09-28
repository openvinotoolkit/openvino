# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(475912)

class TestTFScatterND(CommonTFLayerTest):
    def create_tf_scatternd_placeholder_const_net(self, x_shape, indices, updates, ir_version,
                                                  use_legacy_frontend):
        #
        #   Create Tensorflow model
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'Input')
            tf_indices = tf.constant(indices)
            tf_updates = tf.constant(updates)

            scatter_nd = tf.scatter_nd(tf_indices, tf_updates, tf.shape(x), name="Operation")
            res = tf.add(x, scatter_nd)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data = [
        pytest.param(
            dict(x_shape=[8], indices=[[4], [3], [1], [7]], updates=[9.0, 10.0, 11.0, 12.0]),
            marks=pytest.mark.precommit),
        pytest.param(dict(x_shape=[4, 4, 4], indices=[[0], [2]], updates= \
            [[[5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0],
              [8.0, 8.0, 8.0, 8.0]],
             [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0],
              [4.0, 4.0, 4.0, 4.0]]])),
        pytest.param(dict(x_shape=[2, 2], indices=[[0]], updates=[[5.0, 3.0]])),
        pytest.param(dict(x_shape=[2, 2], indices=[[1, 1]], updates=[5.0])),
        dict(x_shape=[1], indices=[[0]], updates=[3.0]),
        dict(x_shape=[20], indices=[[0], [6], [9], [19], [13]],
             updates=[3.0, 7.0, -12.0, 4.0, -99.0]),
        dict(x_shape=[4, 2], indices=[[1], [2]], updates=[[9.0, 14.0], [-76.0, 0.0]]),
        dict(x_shape=[4, 4, 4], indices=[[0], [1], [3]], updates=[
            [[5.0, 1.0, 5.0, 13.0], [8.0, 6.0, 6.0, 8.0], [7.0, 0.0, 0.0, 7.0],
             [8.0, 8.0, 8.0, 8.0]],
            [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0],
             [9.0, 10.0, 11.0, 12.0]],
            [[5.0, 5.0, 5.0, 5.0], [6.0, 6.0, 6.0, 6.0], [7.0, 7.0, 7.0, 7.0],
             [8.0, 8.0, 8.0, 8.0]]]),
        dict(x_shape=[2, 2, 2], indices=[[1, 1, 1], [0, 1, 0]], updates=[9.0, 6.3]),
        pytest.param(dict(x_shape=[2, 2, 2], indices=[[0, 0], [0, 1]], updates=[[6.7, 9.0], [45.0, 8.3]]),
                     marks=pytest.mark.precommit),
        dict(x_shape=[2, 2, 2], indices=[[1]], updates=[[[6.7, 9.0], [45.0, 8.3]]]),

    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Darwin' and platform.machine() == 'arm64',
                       reason='Ticket - 122716')
    def test_tf_scatter_nd(self, params, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        self._test(*self.create_tf_scatternd_placeholder_const_net(**params, ir_version=ir_version,
                                                                   use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)

class TestTFScatterNDComplex(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'param_real:0' in inputs_info, "Test error: inputs_info must contain `param_real`"
        assert 'param_imag:0' in inputs_info, "Test error: inputs_info must contain `param_imag`"
        updates_shape = inputs_info['param_real:0']
        inputs_data = {}
        inputs_data['param_real:0'] = rng.integers(-10, 10, updates_shape).astype(np.float32)
        inputs_data['param_imag:0'] = rng.integers(-10, 10, updates_shape).astype(np.float32)

        return inputs_data

    def create_tf_scatternd_complex_placeholder_const_net(self, x_shape, indices, updates_shape, indices_type,
                                                          ir_version, use_legacy_frontend):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(tf.float32, updates_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(tf.float32, updates_shape, 'param_imag')

            tf_indices = tf.constant(indices, dtype=indices_type)
            tf_shape = tf.constant(x_shape, dtype=indices_type)

            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            result = tf.scatter_nd(indices=tf_indices, updates=complex, shape=tf_shape, name="Operation")
            tf.raw_ops.Real(input=result)
            tf.raw_ops.Imag(input=result)

            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data = [
        dict(x_shape=[8], indices=[[4], [3], [1], [7]], updates_shape=[4], indices_type=np.int32),
        dict(x_shape=[10], indices=[[0], [2], [4], [6], [8]], updates_shape=[5], indices_type=np.int64),
        dict(x_shape=[5, 5], indices=[[0, 0], [1, 1], [2, 2], [3, 3]], updates_shape=[4], indices_type=np.int64),
        dict(x_shape=[3, 3, 3], indices=[[0, 0, 0], [1, 1, 1], [2, 2, 2]], updates_shape=[3], indices_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_tf_scatter_nd_complex(self, params, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        self._test(*self.create_tf_scatternd_complex_placeholder_const_net(**params, ir_version=ir_version,
                                                                   use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)
