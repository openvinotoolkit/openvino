# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(56234)


class TestUnsortedSegmentMax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data:0' in inputs_info, "Test error: inputs_info must contain `data`"
        assert 'segment_ids:0' in inputs_info, "Test error: inputs_info must contain `segment_ids`"
        data_shape = inputs_info['data:0']
        segment_ids_shape = inputs_info['segment_ids:0']
        inputs_data = {}
        inputs_data['data:0'] = rng.integers(-50, 50, data_shape).astype(self.data_type)
        num_ids = segment_ids_shape[0]
        if num_ids >= self.num_segments_val:
            ids = list(range(self.num_segments_val))
            ids += rng.integers(0, self.num_segments_val, num_ids - self.num_segments_val).tolist()
            rng.shuffle(ids)
        else:
            ids = rng.integers(0, self.num_segments_val, num_ids).tolist()
        inputs_data['segment_ids:0'] = np.array(ids, dtype=self.segment_ids_type)
        return inputs_data

    def create_unsorted_segment_max_net(self, data_shape, segment_ids_shape, num_segments_val, data_type,
                                        segment_ids_type, num_segments_type):
        self.data_type = data_type
        self.segment_ids_type = segment_ids_type
        self.num_segments_val = num_segments_val
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            segment_ids = tf.compat.v1.placeholder(segment_ids_type, segment_ids_shape, 'segment_ids')
            num_segments = tf.constant(num_segments_val, dtype=num_segments_type, shape=[])
            tf.raw_ops.UnsortedSegmentMax(data=data, segment_ids=segment_ids, num_segments=num_segments)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[8], segment_ids_shape=[8], num_segments_val=5),
        dict(data_shape=[10, 4], segment_ids_shape=[10], num_segments_val=5),
        dict(data_shape=[8, 6, 7], segment_ids_shape=[8], num_segments_val=8),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.parametrize("data_type", [np.float32, np.int32])
    @pytest.mark.parametrize("segment_ids_type", [np.int32, np.int64])
    @pytest.mark.parametrize("num_segments_type", [np.int32, np.int64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_unsorted_segment_max_basic(self, params, data_type, segment_ids_type, num_segments_type, ie_device,
                                        precision, ir_version, temp_dir):
        self._test(*self.create_unsorted_segment_max_net(**params,
                                                         data_type=data_type, segment_ids_type=segment_ids_type,
                                                         num_segments_type=num_segments_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_empty_segments = [
        dict(data_shape=[4], segment_ids_shape=[4], num_segments_val=10),
        dict(data_shape=[3, 5], segment_ids_shape=[3], num_segments_val=8),
    ]

    @pytest.mark.parametrize("params", test_data_empty_segments)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_unsorted_segment_max_empty_segments(self, params, ie_device, precision, ir_version, temp_dir):
        if precision == 'FP16':
            pytest.skip("FP16 lowest fill value differs from FP32 for empty segments")
        self._test(*self.create_unsorted_segment_max_net(**params,
                                                         data_type=np.float32, segment_ids_type=np.int32,
                                                         num_segments_type=np.int32),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
