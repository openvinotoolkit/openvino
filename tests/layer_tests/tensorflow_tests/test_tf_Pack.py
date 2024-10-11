# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestPack(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}
        for input_name, input_shape in inputs_info.items():
            inputs_data[input_name] = np.random.randint(-5, 5, input_shape).astype(self.input_type)
        return inputs_data

    def create_pack_net(self, input_shape, input_num, axis, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            inputs = []
            type_map = {
                np.float32: tf.float32,
                np.int32: tf.int32,
            }
            assert input_type in type_map, "Test error: need to update type_map"
            tf_type = type_map[input_type]
            for ind in range(input_num):
                inputs.append(tf.compat.v1.placeholder(tf_type, input_shape, 'input' + str(ind)))
            if axis is not None:
                tf.raw_ops.Pack(values=inputs, axis=axis)
            else:
                tf.raw_ops.Pack(values=inputs)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 4], input_num=2, axis=None, input_type=np.float32),
        dict(input_shape=[3, 1, 2], input_num=3, axis=1, input_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_pack_basic(self, params, ie_device, precision, ir_version, temp_dir,
                        use_legacy_frontend):
        self._test(*self.create_pack_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_negative_axis = [
        dict(input_shape=[2, 4], input_num=2, axis=-1, input_type=np.float32),
        dict(input_shape=[3, 1, 2], input_num=3, axis=-2, input_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_negative_axis)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_pack_negative_axis(self, params, ie_device, precision, ir_version, temp_dir,
                                use_legacy_frontend):
        self._test(*self.create_pack_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestComplexPack(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        inputs_data = {}
        for input_name, input_shape in inputs_info.items():
            inputs_data[input_name] = np.random.randint(-5, 5, input_shape).astype(np.float32)
        return inputs_data

    def create_complex_pack_net(self, input_shape, input_num, axis):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            inputs_real = []
            inputs_imag = []
            for ind in range(input_num):
                input_real = tf.compat.v1.placeholder(tf.float32, input_shape, 'input' + str(ind) + '_real')
                input_imag = tf.compat.v1.placeholder(tf.float32, input_shape, 'input' + str(ind) + '_imag')
                inputs_real.append(input_real)
                inputs_imag.append(input_imag)
            if axis is not None:
                tf.raw_ops.Pack(values=inputs_real + inputs_imag, axis=axis)
            else:
                tf.raw_ops.Pack(values=inputs_real + inputs_imag)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 4], input_num=2, axis=None),
        dict(input_shape=[3, 1, 2], input_num=3, axis=1),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_pack_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                use_legacy_frontend):
        self._test(*self.create_complex_pack_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_negative_axis = [
        dict(input_shape=[2, 4], input_num=2, axis=-1),
        dict(input_shape=[3, 1, 2], input_num=3, axis=-2),
    ]

    @pytest.mark.parametrize("params", test_data_negative_axis)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_pack_negative_axis(self, params, ie_device, precision, ir_version, temp_dir,
                                        use_legacy_frontend):
        self._test(*self.create_complex_pack_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
