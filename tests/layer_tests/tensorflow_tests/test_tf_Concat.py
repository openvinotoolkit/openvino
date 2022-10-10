# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestConcat(CommonTFLayerTest):
    def create_concat_net(self, shape, axis, ir_version, use_new_frontend):
        """
            Tensorflow net               IR net

            Input->Concat        =>      Input->Concat

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            ax = axis

            tf_x_shape = shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)

            # TODO: add concat with const inputs to check fusing (as in ONNX)

            x = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            y = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')  # Input_1 in graph_def

            concat = tf.concat([x, y], axis=ax, name='Operation')
            concat_shape = concat.shape.as_list()

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return tf_net, ref_net

    # TODO: create tests for concat with 1 input and multiple inputs

    test_data_1D = [dict(shape=[1], axis=0),
                    dict(shape=[1], axis=-1)]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_concat_1D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_2D = [dict(shape=[1, 224], axis=0),
                    dict(shape=[1, 224], axis=-1)]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_concat_2D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_3D = [
        dict(shape=[1, 3, 224], axis=0),
        pytest.param(dict(shape=[1, 3, 224], axis=-1), marks=pytest.mark.precommit_tf_fe),
        dict(shape=[1, 3, 224], axis=2)]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_concat_3D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [dict(shape=[1, 3, 100, 224], axis=0),
                    dict(shape=[1, 3, 100, 224], axis=-1),
                    dict(shape=[1, 3, 100, 224], axis=2)]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_concat_4D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_5D = [dict(shape=[1, 3, 50, 100, 224], axis=0),
                    dict(shape=[1, 3, 50, 100, 224], axis=-1),
                    dict(shape=[1, 3, 50, 100, 224], axis=2)]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_concat_5D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                       use_old_api):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
