# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc

from unit_tests.utils.graph import build_graph


class TestSwish(CommonTFLayerTest):
    def create_swish_net(self, shape, ir_version, use_new_frontend):
        """
            Tensorflow net                 IR net

            Input->Swish       =>       Input->Swish

        """

        import tensorflow as tf

        tf.reset_default_graph()

        # Create the graph and model
        with tf.Session() as sess:
            tf_x_shape = shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)
            input = tf.placeholder(tf.float32, tf_x_shape, 'Input')

            tf.nn.swish(input)

            tf.global_variables_initializer()
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
                'Swish': {'kind': 'op', 'type': 'Swish'},
                'Swish_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'Swish'),
                                   ('Swish', 'Swish_data'),
                                   ('Swish_data', 'result')
                                   ])

        return tf_net, ref_net

    test_data_precommit = [
        pytest.param(dict(shape=[1, 3, 50, 100, 224]),
                     marks=pytest.mark.skip(reason="Skipped until fixed"))
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_swish_precommit(self, params, ie_device, precision, ir_version, temp_dir,
                             use_new_frontend):
        self._test(*self.create_swish_net(**params, ir_version=ir_version,
                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)

    test_data = [dict(shape=[1]),
                 dict(shape=[1, 224]),
                 dict(shape=[1, 3, 224]),
                 dict(shape=[1, 3, 100, 224]),
                 dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_swish(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend):
        self._test(*self.create_swish_net(**params, ir_version=ir_version,
                                          use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend)
