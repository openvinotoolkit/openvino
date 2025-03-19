# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestSub(CommonTFLayerTest):
    def create_sub_placeholder_const_net(self, x_shape, y_shape, ir_version, use_legacy_frontend):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'Input')
            constant_value = np.random.randint(-256, 256, y_shape).astype(np.float32)
            if (constant_value == 0).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            sub = tf.subtract(x, y, name="Operation")
            sub_shape = sub.shape.as_list()

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # TODO: implement tests for 2 Consts + Sub

    test_data_1D = [
        dict(x_shape=[1], y_shape=[1]),
        dict(x_shape=[3], y_shape=[3])
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_2D = [
        dict(x_shape=[1, 1], y_shape=[1, 1]),
        dict(x_shape=[1, 3], y_shape=[1, 3]),
        dict(x_shape=[3, 1], y_shape=[3, 1]),
        dict(x_shape=[2, 3], y_shape=[2, 3])
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_3D = [
        dict(x_shape=[1, 1, 1], y_shape=[1, 1, 1]),
        dict(x_shape=[1, 3, 1], y_shape=[1, 3, 1]),
        dict(x_shape=[1, 1, 3], y_shape=[1, 1, 3]),
        pytest.param(dict(x_shape=[1, 3, 224], y_shape=[1, 3, 224]),
                     marks=pytest.mark.xfail(reason="*-19053"))
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1, 3, 1, 1]),
        dict(x_shape=[1, 1, 1, 3], y_shape=[1, 1, 1, 3]),
        dict(x_shape=[1, 3, 222, 224], y_shape=[1, 3, 222, 224])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1, 1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[1, 3, 1, 1, 1]),
        dict(x_shape=[1, 1, 1, 1, 3], y_shape=[1, 1, 1, 1, 3]),
        dict(x_shape=[1, 3, 50, 100, 224], y_shape=[1, 3, 50, 100, 224])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    ###############################################################################################
    #                                                                                             #
    #                                       Broadcast cases                                       #
    #                                                                                             #
    ###############################################################################################

    test_data_broadcast_1D = [
        dict(x_shape=[3], y_shape=[1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_1D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_broadcast_1D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_2D = [
        dict(x_shape=[1, 1], y_shape=[1]),
        dict(x_shape=[1, 3], y_shape=[1]),
        dict(x_shape=[1, 3], y_shape=[3]),
        dict(x_shape=[3, 1], y_shape=[3]),
        dict(x_shape=[3, 1], y_shape=[1, 3, 1, 1]),
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_2D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_broadcast_2D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_3D = [
        dict(x_shape=[1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1], y_shape=[3]),
        dict(x_shape=[1, 3, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[3, 1, 224], y_shape=[1, 3, 224]),
        dict(x_shape=[2, 3, 1], y_shape=[1, 3, 2]),
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_broadcast_3D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_broadcast_3D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 100, 224, 3], y_shape=[3]),
        dict(x_shape=[1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 3, 1, 2], y_shape=[3, 1, 2]),
        dict(x_shape=[1, 3, 1, 2], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 3, 100, 224], y_shape=[1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2], y_shape=[1, 3, 2, 1])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_broadcast_4D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_broadcast_4D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)

    test_data_broadcast_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1, 1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[1, 1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3, 1]),
        pytest.param(dict(x_shape=[1, 3, 1, 1, 2], y_shape=[1, 3, 2]), marks=pytest.mark.precommit),
        dict(x_shape=[1, 3, 5, 1, 2], y_shape=[3, 5, 1, 2]),
        dict(x_shape=[1, 3, 50, 100, 224], y_shape=[1, 1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2, 1], y_shape=[1, 3, 2, 1, 1])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_broadcast_5D)
    @pytest.mark.nightly
    def test_sub_placeholder_const_broadcast_5D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_legacy_frontend):
        self._test(*self.create_sub_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version,
                   temp_dir=temp_dir, use_legacy_frontend=use_legacy_frontend)


class TestComplexSub(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        rng = np.random.default_rng(84821)

        assert 'param_real_x:0' in inputs_info
        assert 'param_imag_x:0' in inputs_info

        assert 'param_real_y:0' in inputs_info
        assert 'param_imag_y:0' in inputs_info

        param_real_shape_x = inputs_info['param_real_x:0']
        param_imag_shape_x = inputs_info['param_imag_x:0']

        param_real_shape_y = inputs_info['param_real_y:0']
        param_imag_shape_y = inputs_info['param_imag_y:0']

        inputs_data = {}
        inputs_data['param_real_x:0'] =  rng.uniform(-10.0, 10.0, param_real_shape_x).astype(np.float32)
        inputs_data['param_imag_x:0'] = rng.uniform(-10.0, 10.0, param_imag_shape_x).astype(np.float32)

        inputs_data['param_real_y:0'] =  rng.uniform(-10.0, 10.0, param_real_shape_y).astype(np.float32)
        inputs_data['param_imag_y:0'] = rng.uniform(-10.0, 10.0, param_imag_shape_y).astype(np.float32)

        return inputs_data

    def create_complex_sub_net(self, x_shape, y_shape, ir_version, use_legacy_frontend):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            param_real_x = tf.compat.v1.placeholder(np.float32, x_shape, 'param_real_x')
            param_imag_x = tf.compat.v1.placeholder(np.float32, x_shape, 'param_imag_x')
            
            param_real_y = tf.compat.v1.placeholder(np.float32, y_shape, 'param_real_y')
            param_imag_y = tf.compat.v1.placeholder(np.float32, y_shape, 'param_imag_y')

            x = tf.raw_ops.Complex(real=param_real_x, imag=param_imag_x)
            y = tf.raw_ops.Complex(real=param_real_y, imag=param_imag_y)

            result = tf.raw_ops.Sub(x=x, y=y, name='Sub')

            tf.raw_ops.Real(input=result)
            tf.raw_ops.Imag(input=result)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    @pytest.mark.parametrize('x_shape, y_shape', [
        [[5, 5], [5]],
        [[4, 10], [4, 1]],
        [[1, 3, 50, 224], [1]],
        [[10, 10, 10], [10, 10, 10]],
    ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_complex_sub(self, x_shape, y_shape, 
                   ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_complex_sub_net(x_shape, y_shape, ir_version=ir_version,
                                          use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
