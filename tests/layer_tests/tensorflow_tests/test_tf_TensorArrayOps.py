# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


def create_tensor_array(data_shape, data_type):
    size = data_shape[0]
    data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
    indices = tf.compat.v1.placeholder(tf.int32, [size], 'indices')
    size_const = tf.constant(size, dtype=tf.int32, shape=[])
    handle, flow = tf.raw_ops.TensorArrayV3(size=size_const, dtype=tf.as_dtype(data_type))
    flow = tf.raw_ops.TensorArrayScatterV3(handle=handle, indices=indices, value=data, flow_in=flow)
    return handle, flow


class TestTensorArraySizeV3(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data' in inputs_info
        assert 'indices' in inputs_info
        data_shape = inputs_info['data']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['data'] = rng.integers(-10, 10, data_shape).astype(self.data_type)
        inputs_data['indices'] = rng.permutation(self.size).astype(np.int32)
        return inputs_data

    def create_tensor_array_size_v3(self, data_shape, data_type):
        size = data_shape[0]
        self.data_type = data_type
        self.size = size
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            handle, flow = create_tensor_array(data_shape, data_type)
            tf.raw_ops.TensorArraySizeV3(handle=handle, flow_in=flow)
            tf.raw_ops.TensorArrayCloseV3(handle=handle)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[5], data_type=np.float32),
        dict(data_shape=[10, 20, 30], data_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_tensor_array_size_v3(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        self._test(*self.create_tensor_array_size_v3(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestTensorArrayReadV3(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data' in inputs_info
        assert 'indices' in inputs_info
        data_shape = inputs_info['data']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['data'] = rng.integers(-10, 10, data_shape).astype(self.data_type)
        inputs_data['index_to_read'] = rng.integers(0, data_shape[0], []).astype(np.int32)
        inputs_data['indices'] = rng.permutation(self.size).astype(np.int32)
        return inputs_data

    def create_tensor_array_read_v3(self, data_shape, data_type):
        size = data_shape[0]
        self.data_type = data_type
        self.size = size
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            handle, flow = create_tensor_array(data_shape, data_type)
            index_to_read = tf.compat.v1.placeholder(tf.int32, [], 'index_to_read')
            tf.raw_ops.TensorArrayReadV3(handle=handle, index=index_to_read, flow_in=flow,
                                         dtype=tf.dtypes.as_dtype(data_type))
            tf.raw_ops.TensorArrayCloseV3(handle=handle)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[6], data_type=np.float32),
        dict(data_shape=[8, 5, 6, 10], data_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_tensor_array_read_v3(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_new_frontend, use_old_api):
        self._test(*self.create_tensor_array_read_v3(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestTensorArrayWriteGatherV3(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data' in inputs_info
        assert 'indices' in inputs_info
        assert 'value_to_write' in inputs_info
        data_shape = inputs_info['data']
        value_shape = inputs_info['value_to_write']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['data'] = rng.integers(-10, 10, data_shape).astype(self.data_type)
        inputs_data['value_to_write'] = rng.integers(-10, 10, value_shape).astype(self.data_type)
        indices_data = rng.permutation(self.size).astype(np.int32)
        inputs_data['indices'] = np.delete(indices_data, np.where(indices_data == self.index_to_write))
        return inputs_data

    def create_tensor_array_write_v3(self, size, data_shape, data_type, index_to_write, indices_to_gather):
        self.data_type = data_type
        self.size = size
        self.index_to_write = index_to_write
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            value_to_write = tf.compat.v1.placeholder(data_type, data_shape[1:], 'value_to_write')
            index_to_write_const = tf.constant(index_to_write, dtype=tf.int32, shape=[])
            indices_to_gather_const = tf.constant(indices_to_gather, dtype=tf.int32, shape=[len(indices_to_gather)])
            data = tf.compat.v1.placeholder(data_type, data_shape, 'data')
            indices = tf.compat.v1.placeholder(tf.int32, [size - 1], 'indices')
            size_const = tf.constant(size, dtype=tf.int32, shape=[])
            handle, flow = tf.raw_ops.TensorArrayV3(size=size_const, dtype=tf.as_dtype(data_type))
            flow = tf.raw_ops.TensorArrayScatterV3(handle=handle, indices=indices, value=data, flow_in=flow)
            flow = tf.raw_ops.TensorArrayWriteV3(handle=handle, index=index_to_write_const,
                                                 value=value_to_write, flow_in=flow)
            tf.raw_ops.TensorArrayGatherV3(handle=handle, indices=indices_to_gather_const, flow_in=flow,
                                           dtype=tf.dtypes.as_dtype(data_type))
            tf.raw_ops.TensorArrayCloseV3(handle=handle)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(size=7, data_shape=[6], data_type=np.float32, index_to_write=3, indices_to_gather=[0, 3, 1]),
        dict(size=10, data_shape=[9, 2, 4], data_type=np.int32, index_to_write=2, indices_to_gather=[2, 1, 4, 3]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_tensor_array_write_v3(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_new_frontend, use_old_api):
        self._test(*self.create_tensor_array_write_v3(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestTensorArrayConcatV3(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'data' in inputs_info
        assert 'indices' in inputs_info
        data_shape = inputs_info['data']
        inputs_data = {}
        rng = np.random.default_rng()
        inputs_data['data'] = rng.integers(-10, 10, data_shape).astype(self.data_type)
        inputs_data['indices'] = rng.permutation(self.size).astype(np.int32)
        return inputs_data

    def create_tensor_array_concat_v3(self, data_shape, data_type):
        size = data_shape[0]
        self.data_type = data_type
        self.size = size
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            handle, flow = create_tensor_array(data_shape, data_type)
            tensor_array_concat_v3 = tf.raw_ops.TensorArrayConcatV3(handle=handle, flow_in=flow,
                                                                    dtype=tf.as_dtype(data_type))
            tf.identity(tensor_array_concat_v3[0], name='values')
            tf.identity(tensor_array_concat_v3[1], name='length')
            tf.raw_ops.TensorArrayCloseV3(handle=handle)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(data_shape=[5, 3, 11, 2], data_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_tensor_array_concat_v3(self, params, ie_device, precision, ir_version, temp_dir,
                                    use_new_frontend, use_old_api):
        self._test(*self.create_tensor_array_concat_v3(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
