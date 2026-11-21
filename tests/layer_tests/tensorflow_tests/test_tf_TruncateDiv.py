# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestTruncateDiv(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        assert 'y:0' in inputs_info
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        # generate x and y to ensure truncation
        inputs_data['x:0'] = np.random.randint(-10, 10, x_shape).astype(self.input_type)
        inputs_data['y:0'] = np.random.randint(1, 10, y_shape).astype(self.input_type)
        return inputs_data

    def create_truncate_div_net(self, input_shape, input_type):
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'x')
            y = tf.compat.v1.placeholder(input_type, input_shape, 'y')
            tf.raw_ops.TruncateDiv(x=x, y=y)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[10, 20], input_type=np.float32),
        dict(input_shape=[8, 5], input_type=np.float32),
        dict(input_shape=[5, 3], input_type=np.int32),
        dict(input_shape=[6, 4], input_type=np.int32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() in ('Linux', 'Darwin') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 126314, 122716')
    def test_truncate_div_basic(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("Requested activation is not supported for integer type or accuracy issue on GPU")
        self._test(*self.create_truncate_div_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir)


class TestTruncateDivExactMultiples(CommonTFLayerTest):
    # Multiples of the divisor must give whole quotients. Computed as a multiplication by the
    # reciprocal, 41 / 41 truncates to 0 in float32, and 3 / 3 truncates to 0 in float16.

    def _prepare_input(self, inputs_info):
        inputs_data = {'x:0': self.x}
        if 'y:0' in inputs_info:
            inputs_data['y:0'] = np.full_like(self.x, self.divisor)
        return inputs_data

    def create_truncate_div_net(self, dtype, divisor, const_divisor):
        self.x = (np.arange(-20, 21) * divisor).astype(dtype)
        self.divisor = divisor
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(dtype, self.x.shape, 'x')
            if const_divisor:
                y = tf.constant(np.array(divisor, dtype=dtype))
            else:
                y = tf.compat.v1.placeholder(dtype, self.x.shape, 'y')
            tf.raw_ops.TruncateDiv(x=x, y=y)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    @pytest.mark.parametrize("dtype,divisor", [(np.float32, 41), (np.float32, 61), (np.float16, 3), (np.float16, 47)])
    @pytest.mark.parametrize("const_divisor", [True, False])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_truncate_div_exact_multiples(self, dtype, divisor, const_divisor, ie_device, precision, ir_version,
                                          temp_dir):
        if ie_device == 'GPU':
            pytest.skip("Requested activation is not supported for integer type or accuracy issue on GPU")
        self._test(*self.create_truncate_div_net(dtype, divisor, const_divisor),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
