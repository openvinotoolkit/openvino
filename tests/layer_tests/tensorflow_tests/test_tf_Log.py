# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest
from unit_tests.utils.graph import build_graph


class TestLog(CommonTFLayerTest):
    disable_input_layout_conversion = True

    def create_log_net(self, shape, ir_version):
        """
            Tensorflow net                 IR net

            Input->Log       =>       Input->Log

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, shape, 'Input')

            tf.math.log(input, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        if check_ir_version(10, None, ir_version):
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': shape, 'kind': 'data'},
                'log': {'kind': 'op', 'type': 'Log'},
                'log_data': {'shape': shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'log'),
                                   ('log', 'log_data'),
                                   ('log_data', 'result')
                                   ])

        return tf_net, ref_net

    test_data_precommit = [
        dict(shape=[3, 2, 3, 7, 6])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_log_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_log_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   kwargs_to_prepare_input={'min_value': 1, 'max_value': 255})

    test_data = [dict(shape=[1]),
                 dict(shape=[2, 5]),
                 dict(shape=[5, 3, 7, 4]),
                 dict(shape=[3, 2, 3, 7, 6])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_log(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_log_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   kwargs_to_prepare_input={'min_value': 1, 'max_value': 255})
