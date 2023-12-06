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
        assert 'param_real' in inputs_info
        assert 'param_imag' in inputs_info
        param_real_shape = inputs_info['param_real']
        param_imag_shape = inputs_info['param_imag']
        inputs_data = {}
        inputs_data['param_real'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
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


class TestComplexRFFT(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'param' in inputs_info
        param_shape = inputs_info['param']
        inputs_data = {}
        inputs_data['param'] = 4 * rng.random(param_shape).astype(np.float32) - 2
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
        assert 'param_real' in inputs_info
        assert 'param_imag' in inputs_info
        param_real_shape = inputs_info['param_real']
        param_imag_shape = inputs_info['param_imag']
        inputs_data = {}
        inputs_data['param_real'] = 4 * rng.random(param_real_shape).astype(np.float32) - 2
        inputs_data['param_imag'] = 4 * rng.random(param_imag_shape).astype(np.float32) - 2
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
