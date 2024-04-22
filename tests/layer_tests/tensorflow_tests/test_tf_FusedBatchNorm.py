# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sys import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestFusedBatchNorm(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}
        x_shape = inputs_info['x:0']
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape)
        scale_shape = inputs_info['scale:0']
        inputs_data['scale:0'] = np.random.randint(-10, 10, scale_shape)
        offset_shape = inputs_info['offset:0']
        inputs_data['offset:0'] = np.random.randint(-10, 10, offset_shape)
        if 'mean:0' in inputs_info:
            mean_shape = inputs_info['mean:0']
            inputs_data['mean:0'] = np.random.randint(-10, 10, mean_shape)
        if 'variance:0' in inputs_info:
            variance_shape = inputs_info['variance:0']
            inputs_data['variance:0'] = np.random.randint(0, 10, variance_shape)
        return inputs_data

    def create_fused_batch_norm_net(self, x_shape, epsilon, exponential_avg_factor, data_format, is_training,
                                    fbn_version, empty_mean_variance=False):
        fbn_dict = {
            "v1": tf.raw_ops.FusedBatchNorm,
            "v2": tf.raw_ops.FusedBatchNormV2,
            "v3": tf.raw_ops.FusedBatchNormV3
        }
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            c_dim = x_shape[-1]
            if data_format == "NCHW" or data_format == "NCDHW":
                c_dim = x_shape[1]
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'x')
            if empty_mean_variance:
                mean = tf.constant([], dtype=tf.float32)
                variance = tf.constant([], dtype=tf.float32)
            else:
                mean = tf.compat.v1.placeholder(tf.float32, [c_dim], 'mean')
                variance = tf.compat.v1.placeholder(tf.float32, [c_dim], 'variance')

            scale = tf.compat.v1.placeholder(tf.float32, [c_dim], 'scale')
            offset = tf.compat.v1.placeholder(tf.float32, [c_dim], 'offset')
            fbn_func = fbn_dict[fbn_version]
            if not is_training:
                # due to limitation in the layer test infrastructure - it finds tensor names for Parameter and Result nodes
                # by get_any_name() that cannot work if some nodes fused to Parameter or Result node have multiple tensor names
                # This issue is tracked in 97192 ticket
                # Now it is worked around by guarding Parameter Node with AddV2
                mean = tf.raw_ops.AddV2(x=mean, y=tf.constant(2.0, dtype=tf.float32))
                variance = tf.raw_ops.AddV2(x=variance, y=tf.constant(2.0, dtype=tf.float32))
            fused_batch_norm = fbn_func(x=x, scale=scale, offset=offset, epsilon=epsilon,
                                        mean=mean, variance=variance,
                                        exponential_avg_factor=exponential_avg_factor, data_format=data_format,
                                        is_training=is_training, name="FusedBatchNorm")
            tf.identity(fused_batch_norm[0], name='y')
            tf.identity(fused_batch_norm[1], name='batch_mean')
            tf.identity(fused_batch_norm[2], name='batch_variance')
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(x_shape=[2, 3, 4, 5], epsilon=0.0001, exponential_avg_factor=1, data_format="NHWC",
             is_training=True,
             fbn_version="v1"),
        dict(x_shape=[2, 3, 4, 5], epsilon=0.0005, exponential_avg_factor=0.3, data_format="NHWC",
             is_training=True,
             fbn_version="v2"),
        dict(x_shape=[3, 2, 1, 5], epsilon=0.00003, exponential_avg_factor=0.7, data_format="NCHW",
             is_training=True,
             fbn_version="v3"),
        dict(x_shape=[3, 4, 2, 5], epsilon=0.0003, exponential_avg_factor=0.0, data_format="NCHW",
             is_training=True,
             fbn_version="v3"),
        dict(x_shape=[2, 3, 4, 5], epsilon=0.0001, exponential_avg_factor=1, data_format="NHWC",
             is_training=False,
             fbn_version="v1"),
        dict(x_shape=[3, 2, 1, 4], epsilon=0.0005, exponential_avg_factor=0.3, data_format="NCHW",
             is_training=False,
             fbn_version="v2"),
        dict(x_shape=[5, 4, 3, 2], epsilon=0.0005, exponential_avg_factor=0.0, data_format="NCHW",
             is_training=False,
             fbn_version="v3"),
        dict(x_shape=[5, 10, 8, 2], epsilon=0.0002, exponential_avg_factor=0.2, data_format="NHWC",
             is_training=True, fbn_version="v3", empty_mean_variance=False),
        # 5D cases
        dict(x_shape=[5, 4, 3, 2, 3], epsilon=0.0005, exponential_avg_factor=0.0, data_format="NCDHW",
             is_training=False, fbn_version="v3"),
        dict(x_shape=[3, 4, 3, 3, 2], epsilon=0.0003, exponential_avg_factor=0.0, data_format="NDHWC",
             is_training=False, fbn_version="v3"),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.skipif(platform == 'darwin', reason="Ticket - 122182")
    def test_fused_batch_norm_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                    use_legacy_frontend):
        self._test(*self.create_fused_batch_norm_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
