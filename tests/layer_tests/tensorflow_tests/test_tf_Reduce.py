# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np

from common.tf_layer_test_class import CommonTFLayerTest


class TestReduceOps(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.random(inputs_dict[input])
        return inputs_dict

    def create_reduce_net(self, shape, operation, keep_dims, axis, ir_version):
        import tensorflow as tf
        fn_mapping = {'sum': tf.reduce_sum,
                      'max': tf.reduce_max,
                      'min': tf.reduce_min,
                      'mean': tf.reduce_mean,
                      'prod': tf.reduce_prod,
                      }
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            shapes = shape.copy()
            if len(shapes) >= 4:
                shapes.append(shapes.pop(1))

            x = tf.compat.v1.placeholder(tf.float32, shapes, 'Input')
            fn_mapping[operation](x, axis=axis, keepdims=keep_dims, name='Operation')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data = []
    for operation in ['sum', 'max', 'prod', 'min', 'mean']:
        test_data.extend([
                          dict(shape=[2, 3], operation=operation, axis=1),
                          dict(shape=[2, 3, 5], operation=operation, axis=-2),
                          dict(shape=[2, 3, 5, 7], operation=operation, axis=2),
                          dict(shape=[2, 3, 5, 7, 9], operation=operation, axis=[2, -1]),
                          ])

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    def test_reduce(self, params, keep_dims, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reduce_net(**params, keep_dims=keep_dims, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_pre_commit = []
    for operation in ['sum', 'max', 'prod', 'min', 'mean']:
        test_data_pre_commit.extend([dict(shape=[2, 3, 5, 7], operation=operation, axis=-2),
                                    ])

    @pytest.mark.parametrize("params", test_data_pre_commit)
    @pytest.mark.parametrize("keep_dims", [False])
    @pytest.mark.precommit
    def test_reduce_precommit(self, params, keep_dims, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_reduce_net(**params, keep_dims=keep_dims, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
