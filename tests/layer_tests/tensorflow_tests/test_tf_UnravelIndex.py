# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestUnravelIndex(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'indices:0' in inputs_info
        indices_shape = inputs_info['indices:0']
        inputs_data = {}
        inputs_data['indices:0'] = np.random.randint(0, self.num_elements, indices_shape).astype(self.input_type)
        return inputs_data

    def create_unravel_index_net(self, input_shape, input_type, dims_value):
        self.input_type = input_type
        self.num_elements = np.prod(dims_value)
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            indices = tf.compat.v1.placeholder(input_type, input_shape, 'indices')
            dims = tf.constant(dims_value, dtype=input_type)
            tf.raw_ops.UnravelIndex(indices=indices, dims=dims)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[10], input_type=np.int32, dims_value=[2, 3, 4]),
        dict(input_shape=[20], input_type=np.int64, dims_value=[5, 5]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_unravel_index_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        self._test(*self.create_unravel_index_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
