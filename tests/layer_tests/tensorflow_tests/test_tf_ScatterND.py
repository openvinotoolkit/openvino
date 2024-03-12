# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
from common.tf_layer_test_class import CommonTFLayerTest
import numpy as np

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
                     marks=pytest.mark.precommit_tf_fe),
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
        
class TestComplexScatterND(CommonTFLayerTest):
    def create_complex_scatter_nd_net(self, x_shape, indices, updates, ir_version,
                                                  use_legacy_frontend):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'Input')
            tf_indices = tf.compat.v1.placeholder(np.int32, [None], 'indices')
            updates_real = tf.compat.v1.placeholder(np.float32, [None], 'updates_real')
            updates_imag = tf.compat.v1.placeholder(np.float32, [None], 'updates_imag')
            updates = tf.raw_ops.Complex(real=updates_real,imag=updates_imag)

            scatter_nd = tf.raw_ops.ScatterNd(indices, updates, tf.shape(x), name="Operation")
            res = tf.add(x_shape, scatter_nd, name="Operation")
            real = tf.raw_ops.Real(input=res)
            img = tf.raw_ops.Imag(input=res)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None
    
    def prepare_input(self, inputs_info):
        rng = np.random.default_rng()

        assert 'indices' in inputs_info
        assert 'updates_real' in inputs_info
        assert 'updates_imag' in inputs_info
        assert 'x_shape' in inputs_info

        indices_shape = inputs_info['indices']
        updates_real_shape = inputs_info['updates_real']
        updates_imag_shape = inputs_info['updates_imag']
        x_shape = inputs_info['x_shape']

        inputs_data = {}
        inputs_data['indices'] = rng.integers(0, 10, indices_shape).astype(np.int32)  # Example range (0, 10), adjust as needed
        inputs_data['updates_real'] = 4 * rng.random(updates_real_shape).astype(np.float32) - 2
        inputs_data['updates_imag'] = 4 * rng.random(updates_imag_shape).astype(np.float32) - 2
        inputs_data['x_shape'] = rng.integers(0, 10, indices_shape).astype(np.int32)

        return inputs_data
    
    test_data_basic = [
        dict(input_shape=[]),
        dict(input_shape=[2]),
        dict(input_shape=[1, 3]),
        dict(input_shape=[2, 3, 4]),
        dict(input_shape=[3, 4, 5, 6]),
    ]

 
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_tf_scatter_nd(self, params, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        self._test(*self.create_complex_scatter_nd_net(**params, ir_version=ir_version,
                                                                   use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, temp_dir=temp_dir, ir_version=ir_version,
                   use_legacy_frontend=use_legacy_frontend, **params)