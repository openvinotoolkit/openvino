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

        #
        #   Create Tensorflow model
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:

            ax = axis

            input_shape_x = shape.copy()
            # reshaping
            if len(input_shape_x) >= 3:
                input_shape_x.append(input_shape_x.pop(1))

            # TODO: add concat with const inputs to check fusing (as in ONNX)

            x = tf.compat.v1.placeholder(tf.float32, input_shape_x, 'Input')
            y = tf.compat.v1.placeholder(tf.float32, input_shape_x, 'Input')  # Input_1 in graph_def

            concat = tf.concat([x, y], axis=ax, name='Operation')
            concat_shape = concat.shape.as_list()

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        # Format axis to positive value
        concat_ax = axis if axis >= 0 else axis + len(shape)
        if len(shape) >= 3:
            # Permute shape to (N,C,...) format and compute correct axis value
            order = [0, len(concat_shape) - 1] + list(range(1, len(concat_shape) - 1))
            concat_shape = [concat_shape[i] for i in order]
            concat_ax = order.index(concat_ax)

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

    test_data_2D = [dict(shape=[1, 224], axis=0),
                    dict(shape=[1, 224], axis=-1)]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_concat_2D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_3D = [pytest.param(dict(shape=[1, 3, 224], axis=0), marks=pytest.mark.xfail(reason="*-19053")),
                    pytest.param(dict(shape=[1, 3, 224], axis=-1), marks=pytest.mark.xfail(reason="*-19053")),
                    pytest.param(dict(shape=[1, 3, 224], axis=2), marks=pytest.mark.xfail(reason="*-19053"))]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_concat_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_4D = [dict(shape=[1, 3, 100, 224], axis=0),
                    dict(shape=[1, 3, 100, 224], axis=-1),
                    dict(shape=[1, 3, 100, 224], axis=2)]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    @pytest.mark.precommit
    def test_concat_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_5D = [dict(shape=[1, 3, 50, 100, 224], axis=0),
                    dict(shape=[1, 3, 50, 100, 224], axis=-1),
                    dict(shape=[1, 3, 50, 100, 224], axis=2)]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_concat_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_concat_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
