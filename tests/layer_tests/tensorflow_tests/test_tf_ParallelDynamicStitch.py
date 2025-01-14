# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestParallelDynamicStitch(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}
        num_elements = 0
        assert len(inputs_info) % 2 == 0, "Number of inputs should be divisible by 2."
        data_input_cnt = len(inputs_info) // 2
        for i in range(1, data_input_cnt + 1):
            indices_in_name = "indices{}:0".format(i)
            assert indices_in_name in inputs_info, "Test error: inputs_info must contain `{}`".format(indices_in_name)
            indices_shape = inputs_info[indices_in_name]
            num_elements = num_elements + np.prod(indices_shape, dtype=int)

        # we support DynamicStitch via decomposition to subgraph with ScatterUpdate op
        # ScatterUpdate has undefined behavior if there are multiple identical indexes
        # indices_array = np.arange(np.random.randint(1, num_elements+1), dtype=np.intc)
        indices_array = np.arange(num_elements, dtype=np.intc)
        np.random.shuffle(indices_array)
        indices_array = np.resize(indices_array, num_elements)

        idx = 0
        for i in range(1, data_input_cnt + 1):
            data_in_name = "data{}:0".format(i)
            indices_in_name = "indices{}:0".format(i)
            assert data_in_name in inputs_info, "Test error: inputs_info must contain `{}`".format(data_in_name)
            data_shape = inputs_info[data_in_name]
            indices_shape = inputs_info[indices_in_name]
            inputs_data[data_in_name] = np.random.randint(-50, 50, data_shape)

            num_elements_i = np.prod(indices_shape, dtype=int)
            inputs_data[indices_in_name] = np.reshape(indices_array[idx:idx + num_elements_i], indices_shape)
            idx = idx + num_elements_i
        return inputs_data

    def create_parallel_dynamic_stitch_net(self, data_input_cnt, shape_of_element, data_type):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            indices = []
            data = []
            data_shape = shape_of_element
            indices_shape = []

            for i in range(1, data_input_cnt + 1):
                indices.append(tf.compat.v1.placeholder(tf.int32, indices_shape, 'indices{}'.format(i)))
                data.append(tf.compat.v1.placeholder(data_type, data_shape, 'data{}'.format(i)))
                data_shape.insert(0, i)
                indices_shape.insert(0, i)
            tf.dynamic_stitch(indices, data)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_input_cnt=1, shape_of_element=[1], data_type=tf.float32),
        dict(data_input_cnt=2, shape_of_element=[2, 2], data_type=tf.float32),
        dict(data_input_cnt=3, shape_of_element=[2, 1, 2], data_type=tf.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_parallel_dynamic_stitch_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        if use_legacy_frontend:
            pytest.skip("DynamicStitch operation is not supported via legacy frontend.")
        if ie_device == 'GPU':
            pytest.skip("GPU error: Invalid constant blob dimensions")
        self._test(*self.create_parallel_dynamic_stitch_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_different_types = [
        dict(data_input_cnt=4, shape_of_element=[3, 2], data_type=tf.float64),
        dict(data_input_cnt=2, shape_of_element=[2, 2, 1], data_type=tf.int64),
        dict(data_input_cnt=3, shape_of_element=[2, 1, 2, 4], data_type=tf.int32),
    ]

    @pytest.mark.parametrize("params", test_data_different_types)
    @pytest.mark.nightly
    def test_parallel_dynamic_stitch_different_types(self, params, ie_device, precision, ir_version, temp_dir,
                                                     use_legacy_frontend):
        if use_legacy_frontend:
            pytest.skip("DynamicStitch operation is not supported via legacy frontend.")
        if ie_device == 'GPU':
            pytest.skip("GPU error: Invalid constant blob dimensions")
        self._test(*self.create_parallel_dynamic_stitch_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
