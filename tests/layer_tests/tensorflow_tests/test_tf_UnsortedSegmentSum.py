# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(23254)


class TestUnsortedSegmentSum(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data:0' in inputs_info, "Test error: inputs_info must contain `data`"
        assert 'segment_ids:0' in inputs_info, "Test error: inputs_info must contain `segment_ids`"
        data_shape = inputs_info['data:0']
        segment_ids_shape = inputs_info['segment_ids:0']
        inputs_data = {}
        inputs_data['data:0'] = rng.integers(-10, 10, data_shape).astype(self.data_type)
        # segment_ids can have negative values
        inputs_data['segment_ids:0'] = rng.integers(-self.num_segments_val, self.num_segments_val,
                                                    segment_ids_shape).astype(self.segment_ids_type)
        return inputs_data

    def create_unsorted_segment_sum_net(self, data_shape, segment_ids_shape, num_segments_val, data_type,
                                        segment_ids_type, num_segments_type):
        self.data_type = data_type
        self.segment_ids_type = segment_ids_type
        self.num_segments_val = num_segments_val
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            segment_ids = tf.compat.v1.placeholder(segment_ids_type, segment_ids_shape, 'segment_ids')
            num_segments = tf.constant(num_segments_val, dtype=num_segments_type, shape=[])
            tf.raw_ops.UnsortedSegmentSum(data=data, segment_ids=segment_ids, num_segments=num_segments)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[8], segment_ids_shape=[8], num_segments_val=10),
        dict(data_shape=[10, 4], segment_ids_shape=[10], num_segments_val=5),
        dict(data_shape=[5, 6, 7], segment_ids_shape=[5], num_segments_val=100),
        dict(data_shape=[5, 2], segment_ids_shape=[5, 2], num_segments_val=5),
        dict(data_shape=[4, 3, 7], segment_ids_shape=[4, 3], num_segments_val=10),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.parametrize("data_type", [np.float32, np.int32])
    @pytest.mark.parametrize("segment_ids_type", [np.int32, np.int64])
    @pytest.mark.parametrize("num_segments_type", [np.int32, np.int64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_unsorted_segment_sum_basic(self, params, data_type, segment_ids_type, num_segments_type, ie_device,
                                        precision, ir_version, temp_dir,
                                        use_legacy_frontend):
        self._test(*self.create_unsorted_segment_sum_net(**params,
                                                         data_type=data_type, segment_ids_type=segment_ids_type,
                                                         num_segments_type=num_segments_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
