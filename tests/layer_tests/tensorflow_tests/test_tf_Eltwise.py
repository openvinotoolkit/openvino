# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestEltwise(CommonTFLayerTest):
    def create_eltwise_net(self, shape, operation, ir_version):
        """
            Tensorflow net                 IR net

            Inputs->Eltwise       =>       Inputs->Eltwise

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

            x = tf.compat.v1.placeholder(tf.float32, shapes, 'Input')
            y = tf.compat.v1.placeholder(tf.float32, shapes, 'Input')     # Input_1 in graph_def

            if operation == 'sum':
                tf.add(x, y, name='Operation')
            elif operation == 'max':
                tf.maximum(x, y, name='Operation')
            elif operation == 'mul':
                tf.multiply(x, y, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return tf_net, ref_net

    test_data = []
    for operation in ['sum', 'max', 'mul']:
        test_data.extend([dict(shape=[1, 224], operation=operation),
                          dict(shape=[1, 224, 224], operation=operation),
                          dict(shape=[1, 3, 224, 224], operation=operation)])

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.nightly
    def test_eltwise(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_eltwise_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_5D = []
    for operation in ['sum', 'max', 'mul']:
        test_data_5D.extend([dict(shape=[1, 3, 224, 224, 224], operation=operation)])

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.precommit
    def test_eltwise_5D_precommit(self, params, ie_device, precision, ir_version, temp_dir):
        if ie_device == 'GPU':
            pytest.skip("5D tensors is not supported on GPU")
        self._test(*self.create_eltwise_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
