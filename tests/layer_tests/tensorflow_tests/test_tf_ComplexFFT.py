# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestComplexFFT(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        return inputs_data

    def create_complex_fft_net(self, input_shape, shift_roll, axis_roll, fft_op):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            shift = tf.constant(shift_roll, dtype=tf.int32)
            axis = tf.constant(axis_roll, dtype=tf.int32)
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)
            roll = tf.raw_ops.Roll(input=complex, shift=shift, axis=axis)
            fft = fft_op(input=roll)
            real = tf.raw_ops.Real(input=fft)
            imag = tf.raw_ops.Imag(input=fft)
            tf.raw_ops.Pack(values=[real, imag], axis=-1)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 50, 2], shift_roll=[10, 1], axis_roll=[-2, -1]),
        dict(input_shape=[4, 20, 3], shift_roll=[2, 10], axis_roll=[0, 1]),
        dict(input_shape=[1, 50, 50, 2], shift_roll=[10, 20], axis_roll=[-2, -1]),
        dict(input_shape=[4, 20, 30, 3], shift_roll=[2, 10], axis_roll=[0, 1]),
        dict(input_shape=[1, 50, 50, 30, 2], shift_roll=[10, 20, 4], axis_roll=[-3, -2, -1]),
        dict(input_shape=[4, 20, 30, 10, 3], shift_roll=[2, 10], axis_roll=[1, 2]),
    ]

    @pytest.mark.parametrize("fft_op", [
        tf.raw_ops.FFT, tf.raw_ops.FFT2D, tf.raw_ops.FFT3D,
        tf.raw_ops.IFFT, tf.raw_ops.IFFT2D, tf.raw_ops.IFFT3D
    ])
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_fft_basic(self, params, fft_op,
                               ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_fft_net(**params, fft_op=fft_op),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api, custom_eps=1e-2)


class TestComplexAbs(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        return inputs_data

    def create_complex_abs_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)
            tf.raw_ops.ComplexAbs(x=complex)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[]),
        dict(input_shape=[2]),
        dict(input_shape=[1, 3]),
        dict(input_shape=[2, 3, 4]),
        dict(input_shape=[3, 4, 5, 6]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_abs_basic(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_abs_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)

class TestComplexMul(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real_1:0' in inputs_info
        assert 'param_imag_1:0' in inputs_info
        assert 'param_real_2:0' in inputs_info
        assert 'param_imag_2:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real_1:0']
        param_imag_shape_1 = inputs_info['param_imag_1:0']
        param_real_shape_2 = inputs_info['param_real_2:0']
        param_imag_shape_2 = inputs_info['param_imag_2:0']
        inputs_data = {}
        inputs_data['param_real_1:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag_1:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        inputs_data['param_real_2:0'] = 4 * rng.random(param_real_shape_2).astype(np.float32) - 2
        inputs_data['param_imag_2:0'] = 4 * rng.random(param_imag_shape_2).astype(np.float32) - 2
        return inputs_data

    def create_complex_mul_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real1 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real_1')
            param_imag1 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag_1')
            param_real2 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real_2')
            param_imag2 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag_2')
            complex1 = tf.raw_ops.Complex(real=param_real1, imag=param_imag1)
            complex2 = tf.raw_ops.Complex(real=param_real2, imag=param_imag2)
            mul = tf.raw_ops.Mul(x=complex1, y=complex2, name="complex_mul")
            real = tf.raw_ops.Real(input=mul)
            img = tf.raw_ops.Imag(input=mul)

    @pytest.mark.parametrize("fft_op", [
        tf.raw_ops.FFT, tf.raw_ops.FFT2D, tf.raw_ops.FFT3D,
        tf.raw_ops.IFFT, tf.raw_ops.IFFT2D, tf.raw_ops.IFFT3D
    ])
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() == 'Linux' and platform.machine() in ['arm', 'armv7l',
                                                                                         'aarch64',
                                                                                         'arm64', 'ARM64'],
                       reason='Ticket - 126314')
    def test_complex_fft_basic(self, params, fft_op,
                               ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_fft_net(**params, fft_op=fft_op),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api, custom_eps=1e-2)


class TestComplexAbs(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real' in inputs_info
        assert 'param_imag' in inputs_info
        param_real_shape = inputs_info['param_real']
        param_imag_shape = inputs_info['param_imag']
        inputs_data = {}
        inputs_data['param_real'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        return inputs_data

    def create_complex_abs_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)
            tf.raw_ops.ComplexAbs(x=complex)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
      #  dict(input_shape=[]),
        dict(input_shape=[2]),
        dict(input_shape=[1, 3]),
        dict(input_shape=[2, 3, 4]),
        dict(input_shape=[3, 4, 5, 6]),
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_mul(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_mul_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)

class TestComplexRFFT(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param:0' in inputs_info
        param_shape = inputs_info['param:0']
        inputs_data = {}
        inputs_data['param:0'] = 4 * rng.random(param_shape).astype(np.float32) - 2
        return inputs_data

    def create_complex_rfft_net(self, input_shape, fft_length, rfft_op):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param = tf.compat.v1.placeholder(np.float32, input_shape, 'param')
            fft_length_const = tf.constant(fft_length, dtype=tf.int32)
            rfft = rfft_op(input=param, fft_length=fft_length_const)
            real = tf.raw_ops.Real(input=rfft)
            imag = tf.raw_ops.Imag(input=rfft)
            tf.raw_ops.Pack(values=[real, imag], axis=-1)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 3, 20], fft_length=[10], rfft_op=tf.raw_ops.RFFT),
        dict(input_shape=[1, 3, 20], fft_length=[20], rfft_op=tf.raw_ops.RFFT),
        dict(input_shape=[1, 3, 20, 40], fft_length=[20, 10], rfft_op=tf.raw_ops.RFFT2D),
        dict(input_shape=[1, 3, 20, 40], fft_length=[10, 40], rfft_op=tf.raw_ops.RFFT2D),
        dict(input_shape=[1, 2, 10, 20, 5], fft_length=[2, 5, 3], rfft_op=tf.raw_ops.RFFT3D),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_rfft_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_rfft_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestComplexIRFFT(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape = inputs_info['param_real:0']
        param_imag_shape = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
        return inputs_data

    def create_complex_irfft_net(self, input_shape, fft_length, irfft_op):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            fft_length_const = tf.constant(fft_length, dtype=tf.int32)
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)
            irfft_op(input=complex, fft_length=fft_length_const)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1, 3, 20], fft_length=[10], irfft_op=tf.raw_ops.IRFFT),
        dict(input_shape=[1, 3, 20], fft_length=[20], irfft_op=tf.raw_ops.IRFFT),
        dict(input_shape=[1, 3, 20, 40], fft_length=[20, 10], irfft_op=tf.raw_ops.IRFFT2D),
        dict(input_shape=[1, 3, 20, 40], fft_length=[10, 40], irfft_op=tf.raw_ops.IRFFT2D),
        pytest.param(dict(input_shape=[1, 10, 20, 30, 5], fft_length=[2, 3, 4], irfft_op=tf.raw_ops.IRFFT3D),
                     marks=pytest.mark.xfail(reason="accuracy-issue-124452"))
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_irfft_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                 use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_irfft_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)



class TestComplexPad(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
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



class TestComplexShape(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_shape_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            out = tf.raw_ops.Shape(input=complex)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[]),
        dict(input_shape=[2, 3]),
        dict(input_shape=[2, 4, 3]),
        dict(input_shape=[2, 5, 3, 6, 8]),
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_shape(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_shape_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestComplexTranspose(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_transpose_net(self, input_shape, perm_value):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            transpose = tf.raw_ops.Transpose(x=complex, perm=perm_value)
            real = tf.raw_ops.Real(input=transpose)
            img = tf.raw_ops.Imag(input=transpose)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 4], perm_value=[1, 0]),
        dict(input_shape=[2, 1, 3, 4], perm_value=[2, 0, 1, 3]),
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_transpose(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_transpose_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestComplexReshape(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_transpose_net(self, input_shape, target_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            transpose = tf.raw_ops.Reshape(tensor=complex, shape=target_shape)
            real = tf.raw_ops.Real(input=transpose)
            img = tf.raw_ops.Imag(input=transpose)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 6], target_shape=[2, 3, 2]),
        dict(input_shape=[2, 4, 5], target_shape=[4, -1, 5]),
        dict(input_shape=[1], target_shape=[])
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_reshape(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_transpose_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)




class TestComplexSqueeze(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_squeeze_net(self, input_shape, axis):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            squeeze = tf.raw_ops.Squeeze(input=complex, axis=axis)
            real = tf.raw_ops.Real(input=squeeze)
            img = tf.raw_ops.Imag(input=squeeze)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[1], axis=[0]),
        dict(input_shape=[3, 1], axis=[]),
        dict(input_shape=[2, 3, 1], axis=[-1]),
        dict(input_shape=[1, 10, 1, 5], axis=[0, 2]),
        dict(input_shape=[1, 22, 1, 1, 10], axis=[0, 2, -2]),
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_squeeze(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_squeeze_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)


class TestComplexStridedSlice(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param_real:0' in inputs_info
        assert 'param_imag:0' in inputs_info
        param_real_shape_1 = inputs_info['param_real:0']
        param_imag_shape_1 = inputs_info['param_imag:0']
        inputs_data = {}
        inputs_data['param_real:0'] = 4 * rng.random(param_real_shape_1).astype(np.float32) - 2
        inputs_data['param_imag:0'] = 4 * rng.random(param_imag_shape_1).astype(np.float32) - 2
        return inputs_data

    def create_complex_strided_slice_net(self, input_shape, begin_value, end_value, strides_value, begin_mask, end_mask,
                                 ellipsis_mask,
                                 new_axis_mask, shrink_axis_mask):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real')
            param_imag = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag')
            complex = tf.raw_ops.Complex(real=param_real, imag=param_imag)

            #transpose = tf.raw_ops.Squeeze(input=complex, axis=axis)
            begin = tf.constant(begin_value, dtype=tf.int32)
            end = tf.constant(end_value, dtype=tf.int32)
            strides = tf.constant(strides_value, dtype=tf.int32)
            strided_slice = tf.raw_ops.StridedSlice(input=complex, begin=begin, end=end, strides=strides, begin_mask=begin_mask,
                                    end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask,
                                    shrink_axis_mask=shrink_axis_mask)
            real = tf.raw_ops.Real(input=strided_slice)
            img = tf.raw_ops.Imag(input=strided_slice)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(input_shape=[2, 5, 4, 3], begin_value=[1, 0, 2, 0], end_value=[2, 5, 4, 2], strides_value=[1, 2, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[1, 5, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 5, 3], strides_value=[1, 2, 3, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=8, shrink_axis_mask=0),
        dict(input_shape=[3, 4, 5, 7], begin_value=[2, 0, 3], end_value=[3, 0, 6], strides_value=[1, 1, 1],
             begin_mask=6, end_mask=6, ellipsis_mask=2, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[1, 4, 7, 2], begin_value=[0, 0, 0], end_value=[0, 6, 0], strides_value=[1, 1, 1],
             begin_mask=6, end_mask=4, ellipsis_mask=1, new_axis_mask=0, shrink_axis_mask=0),
        dict(input_shape=[1, 3, 7, 2], begin_value=[0, 0, 0], end_value=[0, 6, 0], strides_value=[1, 1, 1],
             begin_mask=6, end_mask=4, ellipsis_mask=1, new_axis_mask=8, shrink_axis_mask=0),
        dict(input_shape=[1, 5], begin_value=[0, 0], end_value=[1, 5], strides_value=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[5, 1], begin_value=[0, 0], end_value=[5, 1], strides_value=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[1, 1, 3], begin_value=[0, 0, 0], end_value=[1, 1, 3], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 1], begin_value=[0, 0, 0], end_value=[1, 5, 1], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=4),
        dict(input_shape=[1, 1, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 1, 5, 3], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 1, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 1, 3], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=4),
        dict(input_shape=[1, 5, 5, 1], begin_value=[0, 0, 0, 0], end_value=[1, 5, 1, 1], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=8),
        dict(input_shape=[1, 1, 5, 5, 3], begin_value=[0, 0, 0, 0, 0], end_value=[1, 1, 5, 5, 3],
             strides_value=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=3),
        dict(input_shape=[1, 5, 1, 5, 3], begin_value=[0, 0, 0, 0, 0], end_value=[1, 5, 1, 5, 3],
             strides_value=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=5),
        dict(input_shape=[1, 5, 1, 5, 1], begin_value=[0, 0, 0, 0, 0], end_value=[1, 5, 1, 5, 1],
             strides_value=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=21),

            dict(input_shape=[1, 5], begin_value=[0, 0], end_value=[1, 5], strides_value=[1, 1], begin_mask=0,
                 end_mask=0, ellipsis_mask=0, new_axis_mask=1, shrink_axis_mask=0),
            dict(input_shape=[1, 5], begin_value=[0, 0], end_value=[1, 5], strides_value=[1, 1], begin_mask=0,
                 end_mask=0, ellipsis_mask=0, new_axis_mask=3, shrink_axis_mask=0),
            dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
                 end_mask=0, ellipsis_mask=0, new_axis_mask=3, shrink_axis_mask=0),
            dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
                 end_mask=0, ellipsis_mask=0, new_axis_mask=4, shrink_axis_mask=0),
            dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
                 end_mask=0, ellipsis_mask=0, new_axis_mask=5, shrink_axis_mask=0),
            dict(input_shape=[1, 5, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 5, 3], strides_value=[1, 1, 1, 1],
                 begin_mask=0,
                 end_mask=0, ellipsis_mask=0, new_axis_mask=4, shrink_axis_mask=0),
            dict(input_shape=[1, 5, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 5, 3], strides_value=[1, 1, 1, 1],
                 begin_mask=0,
                 end_mask=0, ellipsis_mask=0, new_axis_mask=2, shrink_axis_mask=0),
            dict(input_shape=[16, 4, 64], begin_value=[0, 0, 0, 0], end_value=[0, 0, 0, 0], strides_value=[1, 1, 1, 1],
                 begin_mask=19,
                 end_mask=19, ellipsis_mask=0, new_axis_mask=12, shrink_axis_mask=0),
    ]
    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_complex_strided_slice(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(
            *self.create_complex_strided_slice_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_new_frontend=use_new_frontend, use_old_api=use_old_api)