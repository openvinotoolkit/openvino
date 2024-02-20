# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc, permute_axis
from openvino.tools.mo.ops.op import PermuteAttrs

from unit_tests.utils.graph import build_graph


class Test_TopK(CommonTFLayerTest):
    @staticmethod
    def create_topK_net(shape, k, ir_version, use_new_frontend):
        """
            Tensorflow net:

                          |-> Values
            Input -> TopK |
                          |-> Indices


            IR net:

                          |-> Values
            Input -> TopK |
                          |-> Indices

        """

        pytest.xfail(reason="95063")

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            shape_net = permute_nchw_to_nhwc(shape)

            input_tensor = tf.compat.v1.placeholder(tf.int32, shape=shape_net, name='Input')
            values, indices = tf.nn.top_k(input_tensor, k=k, sorted=True, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_1D = [
        dict(shape=[15], k=10),
        dict(shape=[15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_TopK_1D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_topK_net(**params, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)

    test_data_2D = [
        dict(shape=[14, 15], k=10),
        dict(shape=[14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_TopK_2D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_topK_net(**params, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)

    test_data_3D = [
        dict(shape=[13, 14, 15], k=10),
        dict(shape=[13, 14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_TopK_3D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_topK_net(**params, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)

    test_data_4D = [
        dict(shape=[12, 13, 14, 15], k=10),
        dict(shape=[12, 13, 14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_TopK_4D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_topK_net(**params, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)

    test_data_5D = [
        dict(shape=[11, 12, 13, 14, 15], k=10),
        dict(shape=[11, 12, 13, 14, 15], k=5),
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_TopK_5D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_topK_net(**params, ir_version=ir_version,
                                         use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
