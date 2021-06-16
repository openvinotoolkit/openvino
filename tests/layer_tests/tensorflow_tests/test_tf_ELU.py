# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from unit_tests.utils.graph import build_graph


class TestELU(CommonTFLayerTest):
    def create_elu_net(self, shape, ir_version):
        """
            Tensorflow net                 IR net

            Input->ELU       =>       Input->ELU

        """

        #
        #   Create Tensorflow model
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:

            shapes = shape.copy()
            # reshaping
            if len(shapes) >= 4:
                shapes.append(shapes.pop(1))
            input = tf.compat.v1.placeholder(tf.float32, shapes, 'Input')

            tf.nn.elu(input, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'ELU': {'kind': 'op', 'type': 'Elu'},
                'ELU_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'ELU'),
                                   ('ELU', 'ELU_data'),
                                   ('ELU_data', 'result')
                                   ])

        return tf_net, ref_net

    test_data_precommit = [dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_elu_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data = [dict(shape=[10, 12]),
                 dict(shape=[8, 10, 12]),
                 dict(shape=[6, 8, 10, 12]),
                 dict(shape=[4, 6, 8, 10, 12])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_elu(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_elu_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
