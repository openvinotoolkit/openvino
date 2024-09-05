# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

OPS = {
    'tf.raw_ops.IRFFT': tf.raw_ops.IRFFT,
    'tf.raw_ops.IRFFT2D': tf.raw_ops.IRFFT2D,
    'tf.raw_ops.IRFFT3D': tf.raw_ops.IRFFT3D,
    'tf.raw_ops.FFT': tf.raw_ops.FFT,
    'tf.raw_ops.FFT2D': tf.raw_ops.FFT2D,
    'tf.raw_ops.FFT3D': tf.raw_ops.FFT3D,
    'tf.raw_ops.IFFT': tf.raw_ops.IFFT,
    'tf.raw_ops.IFFT2D': tf.raw_ops.IFFT2D,
    'tf.raw_ops.IFFT3D': tf.raw_ops.IFFT3D,
    'tf.raw_ops.RFFT': tf.raw_ops.RFFT,
    'tf.raw_ops.RFFT2D': tf.raw_ops.RFFT2D,
    'tf.raw_ops.RFFT3D': tf.raw_ops.RFFT3D
}

rng = np.random.default_rng(346546756)


def aux_prepare_input(inputs_info):
    assert 'param_real:0' in inputs_info
    assert 'param_imag:0' in inputs_info
    param_real_shape = inputs_info['param_real:0']
    param_imag_shape = inputs_info['param_imag:0']
    inputs_data = {}
    inputs_data['param_real:0'] = rng.uniform(-2.0, 2.0, param_real_shape).astype(np.float32)
    inputs_data['param_imag:0'] = rng.uniform(-2.0, 2.0, param_imag_shape).astype(np.float32)
    return inputs_data


class TestComplexFFT(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        return aux_prepare_input(inputs_info)

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

    @pytest.mark.parametrize('fft_op', [
        'tf.raw_ops.FFT', 'tf.raw_ops.FFT2D', 'tf.raw_ops.FFT3D',
        'tf.raw_ops.IFFT', 'tf.raw_ops.IFFT2D', 'tf.raw_ops.IFFT3D'
    ])
    @pytest.mark.parametrize('input_shape, shift_roll, axis_roll', [
        [[1, 50, 2], [10, 1], [-2, -1]],
        [[4, 20, 3], [2, 10], [0, 1]],
        [[1, 50, 50, 2], [10, 20], [-2, -1]],
        [[4, 20, 30, 3], [2, 10], [0, 1]],
        [[1, 50, 50, 30, 2], [10, 20, 4], [-3, -2, -1]],
        [[4, 20, 30, 10, 3], [2, 10], [1, 2]],
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_fft_basic(self, input_shape, shift_roll, axis_roll, fft_op,
                               ie_device, precision, ir_version, temp_dir,
                               use_legacy_frontend):
        if platform.machine() in ['aarch64', 'arm64', 'ARM64'] and fft_op in ['tf.raw_ops.FFT3D', 'tf.raw_ops.IFFT3D']:
            pytest.skip('151532: accuracy error on ARM')
        custom_eps = 1e-2
        if ie_device == 'GPU' and fft_op in ['tf.raw_ops.FFT3D', 'tf.raw_ops.IFFT3D']:
            custom_eps = 2 * 1e-1
        self._test(*self.create_complex_fft_net(input_shape=input_shape, shift_roll=shift_roll,
                                                axis_roll=axis_roll, fft_op=OPS[fft_op]),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=custom_eps)


class TestComplexRFFT(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'param:0' in inputs_info
        param_shape = inputs_info['param:0']
        inputs_data = {}
        inputs_data['param:0'] = rng.uniform(-2.0, 2.0, param_shape).astype(np.float32)
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

    @pytest.mark.parametrize('input_shape, fft_length, rfft_op', [
        [[1, 3, 20], [10], 'tf.raw_ops.RFFT'],
        [[1, 3, 20], [20], 'tf.raw_ops.RFFT'],
        [[1, 3, 20, 40], [20, 10], 'tf.raw_ops.RFFT2D'],
        [[1, 3, 20, 40], [10, 40], 'tf.raw_ops.RFFT2D'],
        [[1, 2, 10, 20, 5], [2, 5, 3], 'tf.raw_ops.RFFT3D']
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_rfft_basic(self, input_shape, fft_length, rfft_op, ie_device, precision, ir_version, temp_dir,
                                use_legacy_frontend):
        custom_eps = None
        if ie_device == 'GPU' and rfft_op in ['tf.raw_ops.RFFT2D']:
            custom_eps = 2 * 1e-2
        self._test(*self.create_complex_rfft_net(input_shape=input_shape, fft_length=fft_length, rfft_op=OPS[rfft_op]),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend, custom_eps=custom_eps)


class TestComplexIRFFT(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        return aux_prepare_input(inputs_info)

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

    @pytest.mark.parametrize('input_shape, fft_length, irfft_op', [
        [[1, 3, 20], [10], 'tf.raw_ops.IRFFT'],
        [[1, 3, 20], [20], 'tf.raw_ops.IRFFT'],
        [[1, 3, 20, 40], [20, 10], 'tf.raw_ops.IRFFT2D'],
        [[1, 3, 20, 40], [10, 40], 'tf.raw_ops.IRFFT2D'],
        [[1, 10, 20, 30, 5], [2, 3, 4], 'tf.raw_ops.IRFFT3D']
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_irfft_basic(self, input_shape, fft_length, irfft_op, ie_device, precision, ir_version, temp_dir,
                                 use_legacy_frontend):
        if ie_device == 'CPU' and input_shape == [1, 10, 20, 30, 5]:
            pytest.skip('124452: accuracy issue on CPU')
        self._test(*self.create_complex_irfft_net(input_shape=input_shape, fft_length=fft_length,
                                                  irfft_op=OPS[irfft_op]),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
