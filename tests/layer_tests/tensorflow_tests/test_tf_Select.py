# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestSelect(CommonTFLayerTest):
    def create_select_net(self, shape_condition, shape_input, ir_version, use_new_frontend):
        """
            Tensorflow net                 IR net

            Condition --|               Condition --|
                        v                           v
            Input_1-> Select            Input_1-> Select
                        ^                           ^
            Input_2-----|               Input_2-----|
        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            # Permute shapes NCHW -> NHWC for TF network creation
            shape_condition_net = permute_nchw_to_nhwc(shape_condition)
            shape_input_net = permute_nchw_to_nhwc(shape_input)

            condition = tf.compat.v1.placeholder(tf.bool, shape_condition_net, 'Input_condition')
            input_1 = tf.compat.v1.placeholder(tf.float32, shape_input_net, 'Input_1')
            input_2 = tf.compat.v1.placeholder(tf.float32, shape_input_net, 'Input_2')

            tf.compat.v1.where(condition, input_1, input_2, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return tf_net, ref_net

    test_data_1D = [dict(shape_condition=[2], shape_input=[2])]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_select_1D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_select_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_2D = [
        pytest.param(dict(shape_condition=[2], shape_input=[2, 3]), marks=pytest.mark.precommit_tf_fe),
        dict(shape_condition=[3, 5], shape_input=[3, 5]),
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_select_2D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_select_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_3D = [
        dict(shape_condition=[3], shape_input=[3, 4, 5]),
        dict(shape_condition=[3, 4, 5], shape_input=[3, 4, 5]),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_select_3D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_select_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [
        dict(shape_condition=[3], shape_input=[3, 4, 5, 6]),
        dict(shape_condition=[3, 4, 5, 6], shape_input=[3, 4, 5, 6]),
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_select_4D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_select_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_5D = [
        dict(shape_condition=[3], shape_input=[3, 4, 5, 6, 7]),
        dict(shape_condition=[3, 4, 5, 6, 7], shape_input=[3, 4, 5, 6, 7]),
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_select_5D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_select_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
