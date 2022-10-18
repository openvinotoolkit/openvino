# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc

# Testing operation Add and AddV2
# Documentation: https://www.tensorflow.org/api_docs/python/tf/raw_ops/Add
#                https://www.tensorflow.org/api_docs/python/tf/raw_ops/AddV2

class TestAdd(CommonTFLayerTest):
    # x_shape - first argument, should be an array (shape)
    # y_shape - second argument, should be an array (shape). Might be None if y_value is passed
    # ir_version - common parameter
    # use_new_frontend - common parameter
    # y_value - fills y_shape by chosen value, uses randint instead. x_value isn't used because it isn't necessary right
    # use_addv2 - force to use AddV2 operation
    def create_add_placeholder_const_net(self, x_shape, y_shape, ir_version, use_new_frontend, y_value = None, use_addv2 = False):
        """
            Tensorflow net                  IR net

            Placeholder->Add       =>       Placeholder->Add/AddV2
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
            if y_value is None:
                constant_value = np.random.randint(-256, 256, tf_y_shape).astype(np.float32)
                if (constant_value == 0).all():
                    # Avoid elimination of the layer from IR
                    constant_value = constant_value + 1
            else:
                constant_value = np.full(tf_y_shape, y_value, dtype=np.float32)
            y = tf.constant(constant_value)

            if use_addv2 == False:
                add = tf.raw_ops.Add(x=x, y=y, name="Operation")
            else:
                add = tf.raw_ops.AddV2(x=x, y=y, name="Operation")
            add_shape = add.shape.as_list()

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # TODO: implement tests for 2 Consts + Add

    test_data_1D = [
        dict(x_shape=[1], y_shape=[1]),
        pytest.param(dict(x_shape=[3], y_shape=[3]), marks=pytest.mark.xfail(reason="19180"))
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_add_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_1D_v2 = test_data_1D.copy() + [
        dict(x_shape=[1], y_shape=[1], y_value=np.inf),
        dict(x_shape=[1], y_shape=[1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_1D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
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
    def test_add_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_2D_v2 = test_data_2D.copy() + [
        dict(x_shape=[1, 3], y_shape=[1, 3], y_value=np.inf),
        dict(x_shape=[1, 3], y_shape=[1, 3], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_2D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
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
    def test_add_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_3D_v2 = test_data_3D.copy() + [
        dict(x_shape=[1, 1, 1], y_shape=[1, 1, 1], y_value=np.inf),
        dict(x_shape=[1, 1, 1], y_shape=[1, 1, 1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_3D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1, 1, 1, 1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1, 3, 1, 1]),
        pytest.param(dict(x_shape=[1, 1, 1, 3], y_shape=[1, 1, 1, 3]),
                     marks=pytest.mark.xfail(reason="*-19180")),
        dict(x_shape=[1, 3, 222, 224], y_shape=[1, 3, 222, 224])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_add_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_4D_v2 = test_data_4D.copy() + [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1, 1, 1, 1], y_value=np.inf),
        dict(x_shape=[1, 1, 1, 1], y_shape=[1, 1, 1, 1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_4D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
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
    def test_add_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_5D_v2 = test_data_5D.copy() + [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1, 1, 1, 1, 1], y_value=np.inf),
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1, 1, 1, 1, 1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_5D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

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
    def test_add_placeholder_const_broadcast_1D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_broadcast_1D_v2 = test_data_broadcast_1D.copy() + [
        dict(x_shape=[3], y_shape=[1], y_value=np.inf),
        dict(x_shape=[3], y_shape=[1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_broadcast_1D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_broadcast_1D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
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
    def test_add_placeholder_const_broadcast_2D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_broadcast_2D_v2 = test_data_broadcast_2D.copy() + [
        dict(x_shape=[1, 1], y_shape=[1], y_value=np.inf),
        dict(x_shape=[1, 1], y_shape=[1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_broadcast_2D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_broadcast_2D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
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

    @pytest.mark.parametrize("params", test_data_broadcast_3D)
    @pytest.mark.nightly
    def test_add_placeholder_const_broadcast_3D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_broadcast_3D_v2 = test_data_broadcast_3D.copy() + [
        dict(x_shape=[1, 1, 1], y_shape=[1], y_value=np.inf),
        dict(x_shape=[1, 1, 1], y_shape=[1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_broadcast_3D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_broadcast_3D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_broadcast_4D = [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 3, 100, 224], y_shape=[3]),
        dict(x_shape=[1, 1, 1, 3], y_shape=[3]),
        pytest.param(dict(x_shape=[1, 3, 1, 1], y_shape=[3, 1]), marks=pytest.mark.precommit_tf_fe),
        dict(x_shape=[1, 2, 1, 3], y_shape=[3, 1, 2]),
        dict(x_shape=[1, 2, 1, 3], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 3, 100, 224], y_shape=[1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2], y_shape=[1, 3, 2, 1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_add_placeholder_const_broadcast_4D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version=ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_broadcast_4D_v2 = test_data_broadcast_4D.copy() + [
        dict(x_shape=[1, 1, 1, 1], y_shape=[1], y_value=np.inf),
        dict(x_shape=[1, 1, 1, 1], y_shape=[1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_broadcast_4D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_broadcast_4D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_broadcast_5D = [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[1, 1]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3]),
        dict(x_shape=[1, 1, 1, 1, 3], y_shape=[3]),
        dict(x_shape=[1, 3, 1, 1, 1], y_shape=[3, 1]),
        dict(x_shape=[1, 2, 1, 1, 3], y_shape=[1, 3, 2]),
        dict(x_shape=[1, 3, 5, 1, 2], y_shape=[5, 3, 2, 1]),
        dict(x_shape=[1, 3, 50, 100, 224], y_shape=[1, 1, 1, 1, 224]),
        dict(x_shape=[2, 3, 1, 2, 1], y_shape=[1, 3, 2, 1, 1])
    ]

    @pytest.mark.parametrize("params", test_data_broadcast_5D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_add_placeholder_const_broadcast_5D(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        # we do not perform transpose in the test in case of new frontend
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend),
                   ie_device, precision,
                   ir_version=ir_version, temp_dir=temp_dir, use_new_frontend=use_new_frontend,
                   use_old_api=use_old_api)

    # Extend test data by specific V2 cases
    test_data_broadcast_5D_v2 = test_data_broadcast_5D.copy() + [
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1], y_value=np.inf),
        dict(x_shape=[1, 1, 1, 1, 1], y_shape=[1], y_value=np.NINF)
    ]
    @pytest.mark.parametrize("params", test_data_broadcast_5D_v2)
    @pytest.mark.nightly
    def test_addv2_placeholder_const_broadcast_5D(self, params, ie_device, precision, ir_version, temp_dir,
                                      use_new_frontend, use_old_api):
        self._test(*self.create_add_placeholder_const_net(**params, ir_version=ir_version,
                                                          use_new_frontend=use_new_frontend, use_addv2 = True),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
