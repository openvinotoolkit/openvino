# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import platform
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import run_in_jenkins

# Testing operation Equal
# Documentation: https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/equal
rng = np.random.default_rng()


class TestTFEqual(CommonTFLayerTest):
    output_type = np.float32
    x_shape = [1]
    y_shape = [1]
    x_value = None
    y_value = None

    # Overload inputs generation to fill dummy Equal input
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            if isinstance(self.x_shape, np.ndarray) or isinstance(self.x_shape, list):
                if not self.x_value is None:
                    inputs_dict[input] = np.full(inputs_dict[input], self.x_value, dtype=self.output_type)
                else:
                    inputs_dict[input] = np.random.randint(-3, 3, inputs_dict[input]).astype(self.output_type)
            else:
                if not self.x_value is None:
                    if not isinstance(self.x_value, list):
                        inputs_dict[input] = self.output_type(self.x_value)
                    else:
                        inputs_dict[input] = np.ndarray(self.x_value, dtype=self.output_type)
                else:
                    raise RuntimeError("x_shape shouldn't be a scalar value, use x_value instead")
        return inputs_dict

    # ir_version - common parameter
    # use_legacy_frontend - common parameter
    # x_shape - first argument, should be an array (shape)
    # output_type - type of operands (numpy types: int32, int64, float16, etc...), different types for operands are not suppoted by TF
    # y_shape - second argument, should be an array (shape). Might be None if y_value is passed
    # x_value - fills x_shape by chosen value, uses randint instead
    # y_value - if y_shape is None - uses y_value as scalar, otherwise fills y_shape by chosen value, uses randint instead
    def create_tf_equal_net(self, ir_version, use_legacy_frontend, x_shape, output_type, y_shape=None, x_value=None,
                            y_value=None):
        self.x_value = x_value
        self.y_value = y_value
        self.output_type = output_type

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            self.x_shape = x_shape.copy() if isinstance(x_shape, list) else x_shape
            self.y_shape = y_shape.copy() if isinstance(y_shape, list) else y_shape

            if self.output_type == np.float16:
                x = tf.compat.v1.placeholder(tf.float16, self.x_shape, 'Input')
            elif self.output_type == np.float32:
                x = tf.compat.v1.placeholder(tf.float32, self.x_shape, 'Input')
            elif self.output_type == np.float64:
                x = tf.compat.v1.placeholder(tf.float64, self.x_shape, 'Input')
            elif self.output_type == np.int32:
                x = tf.compat.v1.placeholder(tf.int32, self.x_shape, 'Input')
            elif self.output_type == np.int64:
                x = tf.compat.v1.placeholder(tf.int64, self.x_shape, 'Input')

            if isinstance(self.y_shape, np.ndarray) or isinstance(self.y_shape, list):
                if not self.y_value is None:
                    constant_value = np.full(self.y_shape, self.y_value, dtype=self.output_type)
                else:
                    constant_value = np.random.randint(-3, 3, self.y_shape).astype(self.output_type)
            else:
                if not self.y_value is None:
                    if not isinstance(self.y_value, list):
                        constant_value = self.output_type(self.y_value)
                    else:
                        constant_value = np.ndarray(self.y_value, dtype=self.output_type)
                else:
                    raise RuntimeError("y_shape shouldn't be a scalar value, use y_value instead")

            y = tf.constant(constant_value)

            tf.equal(x, y)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_int32 = [
        pytest.param(
            dict(x_shape=[2, 3], y_shape=[2, 3]),
            # Comparing shapes with random values (verifies false and possible true)
            marks=pytest.mark.precommit),
        dict(x_shape=[2, 3], y_value=2),  # Comparing shape with scalar value (verifies false and possible true)
        dict(x_shape=[2, 3], y_shape=[2, 3],  # Comparing shapes with same values (verifies true statement)
             x_value=2, y_value=2),
        dict(x_shape=[2, 3], y_value=2,  # Comparing shape with scalar value (verifies true statement)
             x_value=2),
        dict(x_shape=[2, 3, 2], y_shape=[2]),
        # Comparing shapes with different dimensions, random values (false and possible true)
        dict(x_shape=[1, 2, 3, 4], y_shape=[1, 2, 3, 4])
        # Comparing shapes with different dimensions (more than 3, for case with nchw/nhcw), random values (false and possible true)
    ]

    @pytest.mark.parametrize("params", test_data_int32)
    @pytest.mark.nightly
    def test_tf_equal_int32(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_equal_net(**params, ir_version=ir_version,
                                             use_legacy_frontend=use_legacy_frontend, output_type=np.int32),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)

    test_data_int64 = [
        pytest.param(
            dict(x_shape=[2, 3], y_shape=[2, 3]),
            # Comparing shapes with random values (verifies false and possible true)
            marks=pytest.mark.precommit),
        dict(x_shape=[2, 3], y_value=2),  # Comparing shape with scalar value (verifies false and possible true)
        dict(x_shape=[2, 3], y_shape=[2, 3],  # Comparing shapes with same values (verifies true statement)
             x_value=2, y_value=2),
        dict(x_shape=[2, 3], y_value=2,  # Comparing shape with scalar value (verifies true statement)
             x_value=2),
        dict(x_shape=[2, 3, 2], y_shape=[2]),
        # Comparing shapes with different dimensions, random values (false and possible true)
        dict(x_shape=[1, 2, 3, 4], y_shape=[1, 2, 3, 4])
        # Comparing shapes with different dimensions (more than 3, for case with nchw/nhcw), random values (false and possible true)
    ]

    @pytest.mark.parametrize("params", test_data_int64)
    @pytest.mark.nightly
    def test_tf_equal_int64(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_equal_net(**params, ir_version=ir_version,
                                             use_legacy_frontend=use_legacy_frontend, output_type=np.int64),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)

    # Values for checking important corner cases for float values
    # expect:   false   false   false    false   false   false    true    false    true
    x_corner = [1., 1., 1., np.nan, np.nan, np.nan, np.inf, np.inf, -np.inf]
    y_corner = [np.nan, np.inf, -np.inf, np.nan, np.inf, -np.inf, np.inf, -np.inf, -np.inf]

    test_data_float16 = [
        pytest.param(
            dict(x_shape=[2, 3], y_shape=[2, 3]),
            # Comparing shapes with different dimensions, random values (false and possible true)
            marks=pytest.mark.precommit),
        pytest.param(
            dict(x_shape=[9], y_shape=[9],  # Comparing shapes which contains corner cases
                 x_value=x_corner, y_value=y_corner),
            marks=pytest.mark.special_xfail(args={"ie_device": "GPU"}, reason="94234")),
        dict(x_shape=[1, 2, 3, 4], y_shape=[1, 2, 3, 4])
        # Comparing shapes with different dimensions (more than 3, for case with nchw/nhcw), random values (false and possible true)
    ]

    @pytest.mark.parametrize("params", test_data_float16)
    @pytest.mark.nightly
    def test_tf_equal_float16(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_equal_net(**params, ir_version=ir_version,
                                             use_legacy_frontend=use_legacy_frontend, output_type=np.float16),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)

    test_data_float32 = [
        pytest.param(
            dict(x_shape=[2, 3], y_shape=[2, 3]),
            # Comparing shapes with random values (verifies false and possible true)
            marks=pytest.mark.precommit),
        pytest.param(
            dict(x_shape=[9], y_shape=[9],  # Comparing shapes which contains corner cases
                 x_value=x_corner, y_value=y_corner),
            marks=pytest.mark.special_xfail(args={"ie_device": "GPU"}, reason="94234")),
        dict(x_shape=[1, 2, 3, 4], y_shape=[1, 2, 3, 4])
        # Comparing shapes with different dimensions (more than 3, for case with nchw/nhcw), random values (false and possible true)
    ]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_tf_equal_float32(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_equal_net(**params, ir_version=ir_version,
                                             use_legacy_frontend=use_legacy_frontend, output_type=np.float32),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)

    test_data_float64 = [
        pytest.param(
            dict(x_shape=[2, 3], y_shape=[2, 3]),
            # Comparing shapes with different dimensions, random values (false and possible true)
            marks=pytest.mark.precommit),
        pytest.param(
            dict(x_shape=[9], y_shape=[9],  # Comparing shapes which contains corner cases
                 x_value=x_corner, y_value=y_corner),
            marks=pytest.mark.special_xfail(args={"ie_device": "GPU"}, reason="94234")),
        dict(x_shape=[1, 2, 3, 4], y_shape=[1, 2, 3, 4])
        # Comparing shapes with different dimensions (more than 3, for case with nchw/nhcw), random values (false and possible true)
    ]

    @pytest.mark.parametrize("params", test_data_float64)
    @pytest.mark.nightly
    def test_tf_equal_float64(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_tf_equal_net(**params, ir_version=ir_version,
                                             use_legacy_frontend=use_legacy_frontend, output_type=np.float64),
                   ie_device, precision,
                   temp_dir=temp_dir, ir_version=ir_version, use_legacy_frontend=use_legacy_frontend,
                   **params)


class TestEqualStr(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'x:0' in inputs_info
        assert 'y:0' in inputs_info
        x_shape = inputs_info['x:0']
        y_shape = inputs_info['y:0']
        inputs_data = {}
        strings_dictionary = ['UPPER<>CASE SENTENCE', 'lower case\n sentence', ' UppEr LoweR CAse SENtence \t\n',
                              '  some sentence', 'another sentence HERE    ']
        inputs_data['x:0'] = rng.choice(strings_dictionary, x_shape)
        inputs_data['y:0'] = rng.choice(strings_dictionary, y_shape)
        return inputs_data

    def create_equal_net(self, x_shape, y_shape):
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.string, x_shape, 'x')
            y = tf.compat.v1.placeholder(tf.string, y_shape, 'y')
            tf.raw_ops.Equal(x=x, y=y)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('x_shape', [[], [1], [5]])
    @pytest.mark.parametrize('y_shape', [[], [1], [5]])
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() in ('Darwin', 'Linux') and platform.machine() in ['arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'],
                       reason='126314, 132699: Build tokenizers for ARM and MacOS')
    def test_equal_str(self, x_shape, y_shape,
                       ie_device, precision, ir_version, temp_dir,
                       use_legacy_frontend):
        if x_shape == [] and y_shape == []:
            pytest.skip("156746: EqualStr operation outputs 1D tensor for two input scalars")
        if ie_device == 'GPU' or run_in_jenkins():
            pytest.skip("operation extension is not supported on GPU")
        self._test(*self.create_equal_net(x_shape=x_shape, y_shape=y_shape),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestComplexEqual(CommonTFLayerTest):
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

    def create_complex_equal_net(self, input_shape):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            param_real1 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real_1')
            param_imag1 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag_1')
            param_real2 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_real_2')
            param_imag2 = tf.compat.v1.placeholder(np.float32, input_shape, 'param_imag_2')
            complex1 = tf.raw_ops.Complex(real=param_real1, imag=param_imag1)
            complex2 = tf.raw_ops.Complex(real=param_real2, imag=param_imag2)
            tf.raw_ops.Equal(x=complex1, y=complex2, name="complex_equal")
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
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_equal(self, params, ie_device, precision, ir_version, temp_dir,
                           use_legacy_frontend):
        self._test(
            *self.create_complex_equal_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_legacy_frontend=use_legacy_frontend
        )
