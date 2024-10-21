# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import mix_array_with_several_values

rng = np.random.default_rng(3476123)


class TestBucketize(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'input:0' in inputs_info, "inputs_info must contain `input`"
        input_shape = inputs_info['input:0']
        input_type = self.input_type
        boundaries = self.boundaries

        # compute all preferable values that is good to see
        all_prefer_values = []
        boundaries_size = len(boundaries)
        for ind in range(boundaries_size):
            if ind == 0:
                all_prefer_values.append(boundaries[ind] - 1.0)
            else:
                all_prefer_values.append((boundaries[ind - 1] + boundaries[ind]) / 2.0)
            all_prefer_values.append(boundaries[ind])
            if ind == (boundaries_size - 1):
                all_prefer_values.append(boundaries[ind] + 1.0)

        inputs_data = {}
        input_data = rng.choice(200, input_shape).astype(input_type) - 100
        # mix input data with preferable values
        input_data = mix_array_with_several_values(input_data, all_prefer_values, rng)
        inputs_data['input:0'] = input_data
        return inputs_data

    def create_bucketize_net(self, input_shape, input_type, boundaries_size):
        self.input_type = input_type
        # generate boundaries list
        # use wider range for boundaries than input data in order to cover all bucket indices cases
        boundaries = (np.sort(rng.choice(400, boundaries_size, replace=False).astype(np.float32) - 200)).tolist()
        self.boundaries = boundaries

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(input_type, input_shape, 'input')
            tf.raw_ops.Bucketize(input=input, boundaries=boundaries)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize('input_shape', [[], [5], [3, 4], [2, 3, 4]])
    @pytest.mark.parametrize('input_type', [np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize('boundaries_size', [0, 1, 10, 200])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_bucketize(self, input_shape, input_type, boundaries_size,
                       ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        if ie_device == 'GPU' and boundaries_size == 0:
            pytest.skip("152562: sporadic accuracy issue for boundaries_size == 0 on GPU")
        if platform.machine() in ["aarch64", "arm64", "ARM64"] and boundaries_size == 0:
            pytest.skip("149853: segmentation fault or signal 11 for boundaries_size == 0 on CPU")
        self._test(*self.create_bucketize_net(input_shape, input_type, boundaries_size),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
