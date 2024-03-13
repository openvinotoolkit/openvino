# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestSoftsign(CommonTFLayerTest):
    def create_softsign_net(self, shape, ir_version, use_legacy_frontend):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, shape, 'Input')

            tf.nn.softsign(input)

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

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
                      use_legacy_frontend):
        self._test(*self.create_softsign_net(**params, ir_version=ir_version,
                                             use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
