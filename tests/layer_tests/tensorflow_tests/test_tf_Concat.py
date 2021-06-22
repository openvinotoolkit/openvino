# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestConcat(CommonTFLayerTest):
    def create_concat_net(self, shape, axis, ir_version):
        """
            Tensorflow net               IR net

            Input->Concat        =>      Input->Concat

        """
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            ax = axis
            x = tf.compat.v1.placeholder(tf.float32, shape, 'Input')
            y = tf.compat.v1.placeholder(tf.float32, shape, 'Input')  # Input_1 in graph_def

            tf.concat([x, y], axis=ax, name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    # TODO: create tests for concat with 1 input and multiple inputs

    test_data_1D = [dict(shape=[1], axis=0),
                    dict(shape=[1], axis=-1)]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_concat_1D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_2D = [dict(shape=[1, 3], axis=0),
                    dict(shape=[1, 3], axis=-1)]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_concat_2D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_3D = [dict(shape=[1, 3, 5], axis=0),
                    dict(shape=[1, 3, 5], axis=-1),
                    dict(shape=[1, 3, 5], axis=2),
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_concat_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_4D = [dict(shape=[1, 3, 5, 7], axis=0),
                    dict(shape=[1, 3, 5, 7], axis=-1),
                    dict(shape=[1, 3, 5, 7], axis=2)]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_concat_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_5D = [dict(shape=[1, 3, 5, 7, 9], axis=0),
                    dict(shape=[1, 3, 5, 7, 9], axis=-1),
                    dict(shape=[1, 3, 5, 7, 9], axis=2)]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_concat_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
