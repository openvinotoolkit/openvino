# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestPad(CommonTFLayerTest):
    def create_pad_net(self, input_shape, pads_values, const_value, pad_mode, pad_op):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            paddings = tf.constant(pads_values, dtype=tf.int32)
            placeholder = tf.compat.v1.placeholder(tf.float32, input_shape, 'input')
            if pad_op == 'Pad':
                tf.raw_ops.Pad(input=placeholder, paddings=paddings, name='pad')
            elif pad_op == 'PadV2':
                constant_values = tf.constant(const_value, dtype=tf.float32)
                tf.raw_ops.PadV2(input=placeholder, paddings=paddings, constant_values=constant_values, name='pad')
            elif pad_op == 'MirrorPad':
                tf.raw_ops.MirrorPad(input=placeholder, paddings=paddings, mode=pad_mode, name='pad')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
        dict(input_shape=[2, 3], pads_values=[[0, 1], [2, 3]], const_value=None, pad_mode=None, pad_op='Pad'),
        dict(input_shape=[2, 4, 3], pads_values=[[1, 2], [3, 4], [1, 1]], const_value=3, pad_mode=None, pad_op='PadV2'),
        dict(input_shape=[5, 6], pads_values=[[0, 1], [2, 3]], const_value=None, pad_mode='REFLECT',
             pad_op='MirrorPad'),
        dict(input_shape=[4, 6], pads_values=[[2, 1], [3, 1]], const_value=None, pad_mode='SYMMETRIC',
             pad_op='MirrorPad'),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_pad_basic(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_pad_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

class TestComplexPad(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        import numpy as np
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_pad_net(self, input_shape, pads_values):
        import tensorflow as tf
        import numpy as np
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            pad = tf.raw_ops.Pad(input=complex, paddings=pads_values)
            real = tf.raw_ops.Real(input=pad)
            img = tf.raw_ops.Imag(input=pad)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 3], pads_values=[[0, 1], [2, 3]]),
        dict(input_shape=[2, 4, 3], pads_values=[[1, 2], [3, 4], [1, 1]])
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_pad(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_pad_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)
