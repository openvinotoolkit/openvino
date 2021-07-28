# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestSqueeze(CommonTFLayerTest):
    def create_squeeze_net(self, shape, axis, ir_version):
        """
            Tensorflow net                 IR net

            Input->Squeeze       =>       Input->[Permute]->Reshape

        """

        #
        #   Create Tensorflow model
        #

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            x_shape = shape.copy()
            # reshaping
            if len(x_shape) >= 3:
                x_shape.append(x_shape.pop(1))

            x = tf.compat.v1.placeholder(tf.float32, x_shape, 'Input')
            squeeze = tf.squeeze(x, axis=axis, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        unsigned_axis = [ax if ax > -1 else len(x_shape) + ax for ax in axis]
        if not unsigned_axis:
            unsigned_axis = [i for i, dim in enumerate(shape) if dim == 1]

        ref_net = None

        return tf_net, ref_net

    test_data_1D = [
        pytest.param(dict(shape=[1], axis=[]), marks=pytest.mark.xfail(reason="*-18807")),
        pytest.param(dict(shape=[1], axis=[0]), marks=pytest.mark.xfail(reason="*-18859")),
        pytest.param(dict(shape=[1], axis=[-1]), marks=pytest.mark.xfail(reason="*-18859"))
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_squeeze_1D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_2D = [
        pytest.param(dict(shape=[1, 1], axis=[]), marks=pytest.mark.xfail(reason="*-18807")),
        dict(shape=[1, 1], axis=[0]),
        dict(shape=[1, 1], axis=[-1])
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_squeeze_2D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_3D = [
        pytest.param(dict(shape=[1, 1, 3], axis=[]),
                     marks=[pytest.mark.xfail(reason="*-18807"), pytest.mark.xfail(reason="*-19053")]),
        pytest.param(dict(shape=[1, 1, 3], axis=[0]), marks=pytest.mark.xfail(reason="*-19053")),
        pytest.param(dict(shape=[1, 1, 3], axis=[-1]), marks=pytest.mark.xfail(reason="*-19053"))
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_squeeze_3D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_4D = [
        pytest.param(dict(shape=[1, 1, 50, 100], axis=[]), marks=pytest.mark.xfail(reason="*-18807")),
        dict(shape=[1, 1, 50, 100], axis=[0]),
        dict(shape=[1, 1, 50, 100], axis=[-1]),
        dict(shape=[1, 100, 50, 1], axis=[0, 2])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_squeeze_4D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)

    test_data_5D = [
        pytest.param(dict(shape=[1, 1, 50, 100, 224], axis=[]), marks=pytest.mark.xfail(reason="*-18807")),
        pytest.param(dict(shape=[1, 1, 50, 100, 224], axis=[0]), marks=pytest.mark.xfail(reason="*-18879")),
        pytest.param(dict(shape=[1, 1, 50, 100, 224], axis=[-1]), marks=pytest.mark.xfail(reason="*-18879")),
        dict(shape=[1, 224, 1, 100, 1], axis=[0, 3]),
        dict(shape=[1, 224, 1, 100, 1], axis=[0, 1, 3]),
        dict(shape=[1, 224, 1, 1, 100], axis=[0, 1, 2]),
        dict(shape=[1, 224, 1, 1, 1], axis=[0, 1, 2, 3])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.special_xfail(args={'ie_device': 'GPU', 'precision': 'FP16', 'params': {'axis': [0, 3]}},
                               reason="*-19394")
    @pytest.mark.special_xfail(args={'ie_device': 'GPU', 'precision': 'FP16', 'params': {'axis': [0, 1, 3]}},
                               reason="*-19394")
    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_squeeze_5D(self, params, ie_device, precision, ir_version, temp_dir):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir)
