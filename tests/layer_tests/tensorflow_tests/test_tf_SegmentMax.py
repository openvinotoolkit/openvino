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
        inputs_data['data:0'] = rng.integers(-50, 50, data_shape).astype(np.float32)
        inputs_data['segment_ids:0'] = np.sort(rng.integers(0, data_shape[0], segment_ids_shape).astype(np.int32))
        return inputs_data

    def create_segment_max_net(self, data_shape, segment_ids_shape):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(np.float32, data_shape, 'data')
            segment_ids = tf.compat.v1.placeholder(np.int32, segment_ids_shape, 'segment_ids')
            tf.raw_ops.SegmentMax(data=data, segment_ids=segment_ids)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_precommit = [
        dict(data_shape=[10], segment_ids_shape=[10]),
        dict(data_shape=[5, 10], segment_ids_shape=[5]),
        dict(data_shape=[3, 4, 5], segment_ids_shape=[3]),
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_segment_max(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("Operation SegmentMax is not supported on GPU")
        self._test(*self.create_segment_max_net(**params), ie_device, precision, ir_version, temp_dir=temp_dir)
