# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestOneHot(CommonTFLayerTest):
    @staticmethod
    def create_one_hot_net(shape, depth, on_value, off_value, axis, ir_version, use_new_frontend):
        """
            Tensorflow net

            Input -> OneHot

            IR net (can contain Permutes for input/output of OneHot, depending on shapes), all cases are:

            Input (< 3D) -> OneHot

            Input (3D) -> OneHot -> Permute (NHWC -> NCHW)

            Input (> 3D)  -> Permute (NCHW -> NHWC) -> OneHot ->   Permute (NHWC -> NCHW)
        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            # Permute NCHW -> NHWC for TF network creation
            net_shape = permute_nchw_to_nhwc(shape)

            indices = tf.compat.v1.placeholder(tf.int32, shape=net_shape, name='input_indices')

            result = tf.one_hot(indices,
                                depth,
                                on_value,
                                off_value,
                                axis,
                                name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #

        ref_net = None

        return tf_net, ref_net

    test_data_1D = [
        # check for default on/off value, axis params
        dict(shape=[5], depth=7, on_value=None, off_value=None, axis=None),
        dict(shape=[5], depth=7, on_value=2.0, off_value=-1.0, axis=0)]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_OneHot_1D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_one_hot_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_2D = [
        dict(shape=[5, 6], depth=7, on_value=None, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6], depth=7, on_value=5.0, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6], depth=7, on_value=None, off_value=-1.0, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6], depth=7, on_value=None, off_value=None, axis=1),
        # check for default on/off value, axis params
        dict(shape=[5, 6], depth=7, on_value=2.0, off_value=-3.0, axis=0),
        dict(shape=[5, 6], depth=7, on_value=2.0, off_value=-3.0, axis=1),
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_OneHot_2D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_one_hot_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_3D = [
        dict(shape=[5, 6, 7], depth=8, on_value=None, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7], depth=8, on_value=6.0, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7], depth=8, on_value=None, off_value=4.0, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7], depth=8, on_value=None, off_value=None, axis=1),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7], depth=8, on_value=None, off_value=None, axis=0),
        dict(shape=[5, 6, 7], depth=8, on_value=None, off_value=None, axis=1),
        pytest.param(dict(shape=[5, 6, 7], depth=8, on_value=None, off_value=None, axis=2),
                     marks=pytest.mark.precommit_tf_fe),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_OneHot_3D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_one_hot_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [
        dict(shape=[5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=2),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7, 8], depth=9, on_value=5.0, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7, 8], depth=9, on_value=None, off_value=6.0, axis=None),
        # check for default on/off value, axis params
        dict(shape=[5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=0),
        dict(shape=[5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=1),
        dict(shape=[5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=2),
        dict(shape=[5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=3),
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_OneHot_4D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_one_hot_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_5D = [
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=2.0, off_value=None, axis=None),
        # check for default on/off value, axis params
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=4.0, axis=None),
        # check for default on/off value, axis params
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=1),
        # check for default on/off value, axis params
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=0),
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=1),
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=2),
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=3),
        dict(shape=[4, 5, 6, 7, 8], depth=9, on_value=None, off_value=None, axis=4),
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_OneHot_5D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_one_hot_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
