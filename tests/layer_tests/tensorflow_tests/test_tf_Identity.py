# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc

from unit_tests.utils.graph import build_graph


class TestIdentity(CommonTFLayerTest):
    def create_identity_net(self, shape, ir_version, use_new_frontend):
        """
            Tensorflow net                 IR net

            Input->Identity->ReLU     =>     Input->ReLU

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)

            x = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            id = tf.identity(x, name="Operation")
            tf.nn.relu(id, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version) and not use_new_frontend:
            nodes_attributes = {
                'inputX': {'kind': 'op', 'type': 'Parameter'},
                'inputX_data': {'shape': shape, 'kind': 'data'},
                'ReLU': {'kind': 'op', 'type': 'ReLU'},
                'ReLU_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }
            ref_net = build_graph(nodes_attributes,
                                  [('inputX', 'inputX_data'),
                                   ('inputX_data', 'ReLU'),
                                   ('ReLU', 'ReLU_data'),
                                   ('ReLU_data', 'result')
                                   ])

        return tf_net, ref_net

    test_data_precommit = [dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_identity_precommit(self, params, ie_device, precision, ir_version, temp_dir,
                                use_new_frontend, use_old_api):
        self._test(*self.create_identity_net(**params, ir_version=ir_version,
                                             use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data = [dict(shape=[1]),
                 pytest.param(dict(shape=[1, 224]), marks=pytest.mark.precommit_tf_fe),
                 dict(shape=[1, 3, 224]),
                 dict(shape=[1, 3, 100, 224]),
                 dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_identity(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                      use_old_api):
        self._test(*self.create_identity_net(**params, ir_version=ir_version,
                                             use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
