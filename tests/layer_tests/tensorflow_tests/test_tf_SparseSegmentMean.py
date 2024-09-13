# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(475912)


class TestSparseSegmentMean(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'indices:0' in inputs_info
        assert 'values:0' in inputs_info
        assert 'segment_indices:0' in inputs_info

        values_shape = inputs_info['values:0']

        inputs_data = {}
        inputs_data['values:0'] = rng.uniform(-5.0, 5.0, values_shape).astype(self.data_type)

        # generate all possible indices
        all_indices = []
        for row_ind in range(0, self.shape[0]):
            all_indices.append(row_ind)
        inputs_data['indices:0'] = rng.choice(all_indices, self.indices_shape[0], replace=False).astype(self.indices_type)

        segment_ids = []
        for ind in range(0, self.indices_shape[0]):
            segment_ids.append(self.segment_indices_type(rng.integers(0, self.segments_num)))
        inputs_data['segment_indices:0'] = sorted(segment_ids)

        return inputs_data

    def create_sparse_segment_mean(self, data_type, indices_type, segment_indices_type,
                                   shape, indices_shape, segments_num):
        self.data_type = data_type
        self.indices_type = indices_type
        self.segment_indices_type = segment_indices_type
        self.shape = shape
        self.indices_shape = indices_shape
        self.segments_num = segments_num
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            values = tf.compat.v1.placeholder(data_type, shape, 'values')
            indices = tf.compat.v1.placeholder(indices_type, indices_shape, 'indices')
            segments_ids = tf.compat.v1.placeholder(segment_indices_type, indices_shape, 'segment_indices')
            tf.raw_ops.SparseSegmentMean(
                data=values,
                indices=indices,
                segment_ids=segments_ids)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('data_type', [np.float16, np.float32, np.float64])
    @pytest.mark.parametrize('indices_type', [np.int32, np.int64])
    @pytest.mark.parametrize('segment_indices_type', [np.int32, np.int64])
    @pytest.mark.parametrize('shape, indices_shape, segments_num', [
        [[10], [7], 8],
        [[5], [5], 3],
        [[5], [2], 4],
        [[10, 20], [7], 8],
        [[10, 2, 4], [10], 4]
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_sparse_segment_mean(self, data_type, indices_type, segment_indices_type,
                                 shape, indices_shape, segments_num,
                                 ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("GPU error: to_shape was called on a dynamic shape, ticket: 152352")
        self._test(*self.create_sparse_segment_mean(data_type, indices_type, segment_indices_type,
                                                    shape, indices_shape, segments_num),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
