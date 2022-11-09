# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc

from unit_tests.utils.graph import build_graph


class TestSoftsign(CommonTFLayerTest):
    def create_softsign_net(self, shape, ir_version, use_new_frontend):
        """
            Tensorflow net                 IR net

            Input->Softsign       =>       Input->Softsign

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)
            input = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')

            tf.nn.softsign(input)

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
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'Softsign': {'kind': 'op', 'type': 'Softsign'},
                'Softsign_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'Softsign'),
                                   ('Softsign', 'Softsign_data'),
                                   ('Softsign_data', 'result')
                                   ])

        return tf_net, ref_net

    @pytest.mark.parametrize("params",
                             [
                                 dict(shape=[1]),
                                 dict(shape=[1, 224]),
                                 dict(shape=[1, 3, 224]),
                                 dict(shape=[1, 3, 100, 224]),
                                 dict(shape=[1, 3, 50, 100, 224]),
                             ])
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_softsign(self, params, ie_device, precision, ir_version, temp_dir,
                      use_new_frontend, use_old_api):
        self._test(*self.create_softsign_net(**params, ir_version=ir_version,
                                             use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
