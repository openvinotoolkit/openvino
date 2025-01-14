# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestGatherNd(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'params:0' in inputs_info
        assert 'indices:0' in inputs_info
        params_shape = inputs_info['params:0']
        indices_shape = list(inputs_info['indices:0'])
        inputs_data = {}
        inputs_data['params:0'] = np.random.randint(-50, 50, params_shape).astype(self.params_type)
        # generate indices for each slice and concatenate
        indices_slices = []
        for idx in range(self.index_length):
            indices_slice = np.random.randint(0, self.max_indices[idx], indices_shape[:-1] + [1]).astype(
                self.indices_type)
            indices_slices.append(indices_slice)
        indices_slice_rank = len(indices_slices[0].shape)
        inputs_data['indices:0'] = np.concatenate(indices_slices, axis=indices_slice_rank - 1)
        return inputs_data

    def create_gather_nd_net(self, params_shape, params_type, indices_shape, indices_type):
        self.params_type = params_type
        self.indices_type = indices_type
        assert len(indices_shape) > 0, "Incorrect test case: shape of `indices` must be at least rank 1"
        self.index_length = indices_shape[-1]
        assert self.index_length > 0, "The length of indices must be at least 1"
        assert len(params_shape) >= self.index_length, \
            "Incorrect test case: shape of `params` must be at least rank `indices_shape[-1]`"
        self.max_indices = params_shape[:self.index_length]

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            params = tf.compat.v1.placeholder(params_type, params_shape, 'params')
            indices = tf.compat.v1.placeholder(indices_type, indices_shape, 'indices')
            tf.raw_ops.GatherNd(params=params, indices=indices)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_precommit = [
        dict(params_shape=[2, 3], params_type=np.float32, indices_shape=[2], indices_type=np.int32),
        dict(params_shape=[4, 6, 2], params_type=np.float32, indices_shape=[3, 2], indices_type=np.int32),
        dict(params_shape=[2, 4, 5, 3], params_type=np.int32, indices_shape=[1, 4, 3], indices_type=np.int64),

    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_gather_nd_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_gather_nd_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
