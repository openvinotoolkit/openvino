# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestRsqrt(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.randint(1, 256, inputs_dict[input]).astype(np.float32)
        return inputs_dict

    def create_rsqrt_net(self, shape, ir_version):
        """
            Tensorflow net                 IR net

            Input->Rsqrt       =>       Input->Power

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
            if len(shapes) >= 3:
                shapes.append(shapes.pop(1))
            input = tf.compat.v1.placeholder(tf.float32, shapes, 'Input')

            tf.math.rsqrt(input, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return tf_net, ref_net

    test_data_precommit = [dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_rsqrt_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_rsqrt_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data = [dict(shape=[1]),
                 dict(shape=[1, 224]),
                 pytest.param(dict(shape=[1, 3, 224]), marks=pytest.mark.xfail(reason="*-19053")),
                 dict(shape=[1, 3, 100, 224]),
                 dict(shape=[1, 3, 50, 100, 224])]

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_rsqrt(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_rsqrt_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
