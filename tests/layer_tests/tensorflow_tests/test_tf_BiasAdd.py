# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestBiasAdd(CommonTFLayerTest):
    def create_bias_add_placeholder_const_net(self, shape, ir_version, use_legacy_frontend, output_type=tf.float32):
        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()
            tf_y_shape = tf_x_shape[-1:]

            x = tf.compat.v1.placeholder(output_type, tf_x_shape, 'Input')
            constant_value = np.random.randint(0, 1, tf_y_shape).astype(output_type.as_numpy_dtype())
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            tf.nn.bias_add(x, y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    def create_bias_add_2_consts_net(self, shape, ir_version, use_legacy_frontend, output_type=tf.float32):
        tf.compat.v1.reset_default_graph()
        tf_concat_axis = -1

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()
            tf_y_shape = tf_x_shape[-1:]

            constant_value_x = np.random.randint(-256, 256, tf_x_shape).astype(output_type.as_numpy_dtype())
            x = tf.constant(constant_value_x)
            constant_value_y = np.random.randint(-256, 256, tf_y_shape).astype(output_type.as_numpy_dtype())
            y = tf.constant(constant_value_y)

            add = tf.nn.bias_add(x, y, name="Operation")

            placeholder = tf.compat.v1.placeholder(output_type, tf_x_shape,
                                                   'Input')  # Input_1 in graph_def

            concat = tf.concat([placeholder, add], axis=tf_concat_axis, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_2D = [
        dict(shape=[1, 1]),
        dict(shape=[1, 224])
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_bias_add_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_3D = [
        pytest.param(dict(shape=[1, 1, 224]), marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(shape=[1, 3, 224]), marks=pytest.mark.xfail(reason="*-19053"))
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_bias_add_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_4D = [
        dict(shape=[1, 1, 100, 224]),
        pytest.param(dict(shape=[1, 3, 100, 224]), marks=pytest.mark.precommit),
        pytest.param(dict(shape=[1, 3, 100, 224], output_type=tf.float16), marks=pytest.mark.precommit)
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bias_add_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        if ie_device == 'CPU':
            pytest.skip('155622: OpenVINO runtime timeout on CPU')
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_5D = [
        dict(shape=[1, 1, 50, 100, 224]),
        dict(shape=[1, 3, 220, 222, 224])
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_bias_add_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                           use_legacy_frontend):
        self._test(*self.create_bias_add_placeholder_const_net(**params, ir_version=ir_version,
                                                               use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_bias_add_2_consts_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_bias_add_2_consts_net(**params, ir_version=ir_version,
                                                      use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


class TestComplexBiasAdd(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng()
        assert 'x_real:0' in inputs_info
        assert 'x_imag:0' in inputs_info
        assert 'y_real:0' in inputs_info
        assert 'y_imag:0' in inputs_info
        x_shape = inputs_info['x_real:0']
        y_shape = inputs_info['y_real:0']
        inputs_data = {}

        inputs_data['x_real:0'] = 4 * rng.random(x_shape).astype(np.float64) - 2
        inputs_data['x_imag:0'] = 4 * rng.random(x_shape).astype(np.float64) - 2

        inputs_data['y_real:0'] = 4 * rng.random(y_shape).astype(np.float64) - 2
        inputs_data['y_imag:0'] = 4 * rng.random(y_shape).astype(np.float64) - 2

        return inputs_data

    def create_complex_bias_add_net(self, input_shape, bias_shape, data_format, ir_version, use_legacy_frontend, output_type=tf.float32):
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            x_real = tf.compat.v1.placeholder(output_type, input_shape, 'x_real')
            x_imag = tf.compat.v1.placeholder(output_type, input_shape, 'x_imag')

            y_real = tf.compat.v1.placeholder(output_type, bias_shape, 'y_real')
            y_imag = tf.compat.v1.placeholder(output_type, bias_shape, 'y_imag')

            complex_input = tf.complex(x_real, x_imag)
            complex_bias = tf.complex(y_real, y_imag)

            result = tf.raw_ops.BiasAdd(value=complex_input, bias=complex_bias,data_format=data_format,name="ComplexBiasAdd")
            real = tf.raw_ops.Real(input=result)
            img = tf.raw_ops.Imag(input=result)

            tf_net = sess.graph_def

        return tf_net, None

    test_data_2D = [
        dict(input_shape=[1, 1], bias_shape=[1], data_format="NHWC"),
        dict(input_shape=[3, 2, 7], bias_shape=[7], data_format="NHWC"),
        dict(input_shape=[3, 2, 7, 10], bias_shape=[2], data_format="NCHW"),
        dict(input_shape=[7, 6, 4, 5], bias_shape=[6], data_format="NCHW"),
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_bias_add(self, params, ie_device, precision, ir_version, temp_dir,
                              use_legacy_frontend):
        self._test(*self.create_complex_bias_add_net(**params, ir_version=ir_version,
                                                     use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)