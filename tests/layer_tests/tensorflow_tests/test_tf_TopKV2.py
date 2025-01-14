# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

rng = np.random.default_rng(235436)


class TestTopKV2(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        # generate elements so that the input tensor may contain repeating elements
        assert 'input:0' in inputs_info, "Test error: inputs_info must contain `x`"
        input_shape = inputs_info['input:0']
        inputs_data = {}
        # need to generate data with repeated elements so the range is narrow from 30 elements
        if np.issubdtype(self.input_type, np.floating):
            inputs_data['input:0'] = rng.integers(-15, 15, input_shape).astype(self.input_type)
        elif np.issubdtype(self.input_type, np.signedinteger):
            inputs_data['input:0'] = rng.integers(-15, 15, input_shape).astype(self.input_type)
        else:
            inputs_data['input:0'] = rng.integers(0, 30, input_shape).astype(self.input_type)
        return inputs_data

    def create_topk_v2_net(self, input_shape, input_type, k, k_type, sorted, index_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            k = tf.constant(k, dtype=k_type, shape=[])
            topk = tf.raw_ops.TopKV2(input=input, k=k, sorted=sorted, index_type=index_type)
            tf.identity(topk[0], name='topk_values')
            tf.identity(topk[1], name='unique_indices')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[30], [4, 30], [4, 2, 70]])
    @pytest.mark.parametrize('input_type', [np.float16, np.float32, np.float64,
                                            np.int32, np.uint8, np.int16, np.int8, np.int64,
                                            np.uint16, np.uint32, np.uint64])
    @pytest.mark.parametrize('k', [5, 10, 20])
    @pytest.mark.parametrize('k_type', [np.int16, np.int32, np.int64])
    @pytest.mark.parametrize('sorted', [True,
                                        # TensorFlow has undefined behaviour (or poorly described in the spec)
                                        # for sorted = False
                                        ])
    @pytest.mark.parametrize('index_type', [np.int16, np.int32, np.int64])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_topk_v2(self, input_shape, input_type, k, k_type, sorted, index_type,
                     ie_device, precision, ir_version,
                     temp_dir, use_legacy_frontend):
        if platform.machine() in ['arm', 'armv7l', 'aarch64', 'arm64', 'ARM64'] and \
                input_type in [np.int32, np.uint8, np.int16, np.int8, np.int64,
                               np.uint16, np.uint32, np.uint64]:
            pytest.skip('156588: Segmenation fault or OpenVINO hangs for TopKV2 on ARM')
        self._test(*self.create_topk_v2_net(input_shape, input_type, k, k_type, sorted, index_type),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
