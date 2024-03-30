# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestMul(CommonTFLayerTest):
    def create_mul_placeholder_const_net(self, x_shape, y_shape):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'Input')
            constant_value = np.random.randint(-255, 255, y_shape).astype(np.float32)
            if (constant_value == 1).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            tf.multiply(x, y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    # TODO: implement tests for 2 Consts + Mul

    test_data_1D = [
        dict(x_shape=[1], y_shape=[1]),
        pytest.param(dict(x_shape=[3], y_shape=[3]), marks=pytest.mark.xfail(reason="*-19180"))
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_2D = [
        dict(x_shape=[1, 1], y_shape=[1, 1]),
        dict(x_shape=[1, 3], y_shape=[1, 3]),
        pytest.param(dict(x_shape=[3, 1], y_shape=[3, 1]),
                     marks=pytest.mark.xfail(reason="*-19180")),
        dict(x_shape=[2, 3], y_shape=[2, 3])
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_3D = [
        dict(x_shape=[1, 1, 1], y_shape=[1, 1, 1]),
        pytest.param(dict(x_shape=[1, 3, 1], y_shape=[1, 3, 1]),
                     marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(x_shape=[1, 1, 3], y_shape=[1, 1, 3]),
                     marks=[pytest.mark.xfail(reason="*-19053"),
                            pytest.mark.xfail(reason="*-18830")]),
        pytest.param(dict(x_shape=[1, 3, 224], y_shape=[1, 3, 224]),
                     marks=pytest.mark.xfail(reason="*-19053"))
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1, 3, 1, 1]),
        pytest.param(dict(x_shape=[1, 1, 1, 3], y_shape=[1, 1, 1, 3]),
                     marks=pytest.mark.xfail(reason="*-19180")),
        dict(x_shape=[1, 3, 222, 224], y_shape=[1, 3, 222, 224])
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1, 1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[1, 3, 1, 1, 1]),
        pytest.param(dict(x_shape=[1, 1, 1, 1, 3], y_shape=[1, 1, 1, 1, 3]),
                     marks=pytest.mark.xfail(reason="*-19180")),
        dict(x_shape=[1, 3, 50, 100, 224], y_shape=[1, 3, 50, 100, 224])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    ###############################################################################################
    #                                                                                             #
    #                                       Broadcast cases                                       #
    #                                                                                             #
    ###############################################################################################

    test_data_broadcast_1D = [  # Power
        dict(x_shape=[3], y_shape=[1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_1D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_1D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_2D = [
        dict(x_shape=[1, 1], y_shape=[1]),
        dict(x_shape=[1, 3], y_shape=[1]),
        dict(x_shape=[1, 3], y_shape=[3]),
        dict(x_shape=[3, 1], y_shape=[3]),
        pytest.param(dict(x_shape=[3, 1], y_shape=[1, 3, 1, 1]),
                     marks=pytest.mark.xfail(reason="*-19051"))
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_2D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_2D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_3D = [
        dict(x_shape=[1, 1, 1], y_shape=[1]),
        pytest.param(dict(x_shape=[1, 3, 1], y_shape=[1]),
                     marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(x_shape=[1, 3, 1], y_shape=[3]),
                     marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(x_shape=[1, 3, 1], y_shape=[3, 1]),
                     marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(x_shape=[1, 1, 1], y_shape=[3, 1]),
                     marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(x_shape=[3, 1, 224], y_shape=[1, 3, 224]),
                     marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(x_shape=[2, 3, 1], y_shape=[1, 3, 2]),
                     marks=pytest.mark.xfail(reason="*-19053")),
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_broadcast_3D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_3D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 100, 224, 3], y_shape=[3]),
        dict(x_shape=[1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 3, 1, 2], y_shape=[3, 1, 2]),
        dict(x_shape=[1, 2, 1, 2], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 3, 100, 224], y_shape=[1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2], y_shape=[1, 3, 2, 1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mul_placeholder_const_broadcast_4D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[1, 1]),
        pytest.param(dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3]), marks=pytest.mark.precommit),
        dict(x_shape=[1, 1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 3, 1, 1, 2], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 5, 3, 1, 2], y_shape=[5, 3, 2, 1]),
        dict(x_shape=[1, 3, 50, 100, 224], y_shape=[1, 1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2, 1], y_shape=[1, 3, 2, 1, 1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_5D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_5D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_mul_placeholder_const_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)


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
        import tensorflow as tf
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
    def test_complex_mul(self, params, ie_device, precision, ir_version, temp_dir,
                         use_legacy_frontend):
        self._test(
            *self.create_complex_mul_net(**params),
            ie_device, precision, ir_version, temp_dir=temp_dir,
            use_legacy_frontend=use_legacy_frontend)
