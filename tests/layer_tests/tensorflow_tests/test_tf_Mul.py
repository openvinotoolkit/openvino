# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestMul(CommonTFLayerTest):
    def create_mul_placeholder_const_net(self, x_shape, y_shape, ir_version, use_new_frontend):
        """
            Tensorflow net                  IR net

            Placeholder->Mul       =>       Placeholder->Multiply
                         /                               /
            Const-------/                   Const-------/

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = x_shape.copy()
            tf_y_shape = y_shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)
            tf_y_shape = permute_nchw_to_nhwc(tf_y_shape, use_new_frontend)

            x = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            constant_value = np.random.randint(-255, 255, tf_y_shape).astype(np.float32)
            if (constant_value == 1).all():
                # Avoid elimination of the layer from IR
                constant_value = constant_value + 1
            y = tf.constant(constant_value)

            mul = tf.multiply(x, y, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return tf_net, ref_net

    # TODO: implement tests for 2 Consts + Mul

    test_data_1D = [
        dict(x_shape=[1], y_shape=[1]),
        pytest.param(dict(x_shape=[3], y_shape=[3]), marks=pytest.mark.xfail(reason="*-19180"))
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
                                      use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
                                      use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
                                      use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
                                      use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_broadcast_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 3, 100, 224], y_shape=[3]),
        dict(x_shape=[1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 2, 1, 3], y_shape=[3, 1, 2]),
        dict(x_shape=[1, 2, 1, 3], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 3, 100, 224], y_shape=[1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2], y_shape=[1, 3, 2, 1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_mul_placeholder_const_broadcast_4D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_broadcast_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[1, 1]),
        pytest.param(dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3]), marks=pytest.mark.precommit_tf_fe),
        dict(x_shape=[1, 1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 3, 1, 1, 2], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 3, 5, 1, 2], y_shape=[5, 3, 2, 1]),
        dict(x_shape=[1, 3, 50, 100, 224], y_shape=[1, 1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2, 1], y_shape=[1, 3, 2, 1, 1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_5D)
    @pytest.mark.nightly
    def test_mul_placeholder_const_broadcast_5D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_mul_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
