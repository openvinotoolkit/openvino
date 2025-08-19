# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng()

class TestSegmentMax(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data:0' in inputs_info
        assert 'segment_ids:0' in inputs_info
        data_shape = inputs_info['data:0']
        segment_ids_shape = inputs_info['segment_ids:0']
        inputs_data = {}
        data_type = getattr(self, 'data_type', np.float32)
        segment_ids_type = getattr(self, 'segment_ids_type', np.int32)

        inputs_data['data:0'] = rng.integers(-50, 50, data_shape).astype(data_type)
        inputs_data['segment_ids:0'] = np.sort(rng.integers(0, data_shape[0], segment_ids_shape).astype(segment_ids_type))
        return inputs_data

    def create_segment_max_net(self, data_shape, segment_ids_shape, data_type=np.float32, segment_ids_type=np.int32):
        self.data_type = data_type
        self.segment_ids_type = segment_ids_type
        
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            segment_ids = tf.compat.v1.placeholder(segment_ids_type, segment_ids_shape, 'segment_ids')
            tf.raw_ops.SegmentMax(data=data, segment_ids=segment_ids)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_precommit = [
        dict(data_shape=[10], segment_ids_shape=[10], data_type=np.float32, segment_ids_type=np.int32),
        dict(data_shape=[5, 10], segment_ids_shape=[5], data_type=np.float64, segment_ids_type=np.int32),
        dict(data_shape=[3, 4, 5], segment_ids_shape=[3], data_type=np.int32, segment_ids_type=np.int64),
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_segment_max(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("Operation SegmentMax is not supported on GPU")
        self._test(*self.create_segment_max_net(**params), ie_device, precision, ir_version, temp_dir=temp_dir)

    def create_segment_max_with_missing_ids_net(self, data_shape):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(np.float32, data_shape, 'data')
            segment_shape = [data_shape[0]]
            segment_ids = tf.compat.v1.placeholder(np.int32, segment_shape, 'segment_ids')
            tf.raw_ops.SegmentMax(data=data, segment_ids=segment_ids)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    def _prepare_missing_segments_input(self, inputs_info):
        assert 'data:0' in inputs_info
        assert 'segment_ids:0' in inputs_info
        data_shape = inputs_info['data:0']
        segment_ids_shape = inputs_info['segment_ids:0']
        inputs_data = {}
        inputs_data['data:0'] = rng.integers(-50, 50, data_shape).astype(np.float32)

        segments = np.zeros(segment_ids_shape, dtype=np.int32)

        # Distribute data into segments, leaving some segments empty
        segments[:data_shape[0]//3] = 0  # First third to segment 0
        segments[data_shape[0]//3:2*data_shape[0]//3] = 2  # Second third to segment 2
        segments[2*data_shape[0]//3:] = 4  # Last third to segment 4

        inputs_data['segment_ids:0'] = segments
        return inputs_data

    test_missing_segments_data = [
        dict(data_shape=[15]),
        dict(data_shape=[20]),
    ]

    @pytest.mark.parametrize("params", test_missing_segments_data)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_segment_max_with_missing_ids(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("Operation SegmentMax is not supported on GPU")
        
        original_prepare_input = self._prepare_input
        self._prepare_input = self._prepare_missing_segments_input
        self._test(*self.create_segment_max_with_missing_ids_net(**params), ie_device, precision, ir_version, temp_dir=temp_dir)
        self._prepare_input = original_prepare_input
