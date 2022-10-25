# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest
from common.utils.tf_utils import permute_nchw_to_nhwc


class TestSqueeze(CommonTFLayerTest):
    def create_squeeze_net(self, shape, axis, ir_version, use_new_frontend):
        """
            Tensorflow net                 IR net

            Input->Squeeze       =>       Input->[Permute]->Reshape

        """

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            tf_x_shape = shape.copy()

            tf_x_shape = permute_nchw_to_nhwc(tf_x_shape, use_new_frontend)

            x = tf.compat.v1.placeholder(tf.float32, tf_x_shape, 'Input')
            squeeze = tf.squeeze(x, axis=axis, name="Operation")

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        #
        #   Create reference IR net
        #   Please, specify 'type': 'Input' for input node
        #   Moreover, do not forget to validate ALL layer attributes!!!
        #

        ref_net = None

        return tf_net, ref_net

    test_data_1D = [
        pytest.param(dict(shape=[1], axis=[])),
        pytest.param(dict(shape=[1], axis=[0])),
        pytest.param(dict(shape=[1], axis=[-1]))
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_squeeze_1D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_2D = [
        dict(shape=[1, 1], axis=[]),
        dict(shape=[1, 1], axis=[0]),
        pytest.param(dict(shape=[1, 1], axis=[-1]), marks=pytest.mark.precommit_tf_fe)
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_squeeze_2D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_3D = [
        dict(shape=[1, 1, 3], axis=[]),
        dict(shape=[1, 1, 3], axis=[0]),
        dict(shape=[1, 1, 3], axis=[-1])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_squeeze_3D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_4D = [
        dict(shape=[1, 1, 50, 100], axis=[]),
        dict(shape=[1, 1, 50, 100], axis=[0]),
        dict(shape=[1, 1, 50, 100], axis=[-1]),
        dict(shape=[1, 100, 50, 1], axis=[0, 2])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_squeeze_4D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_5D = [
        dict(shape=[1, 1, 50, 100, 224], axis=[]),
        dict(shape=[1, 1, 50, 100, 224], axis=[0]),
        dict(shape=[1, 1, 50, 100, 224], axis=[-1]),
        dict(shape=[1, 224, 1, 100, 1], axis=[0, 3]),
        dict(shape=[1, 224, 1, 100, 1], axis=[0, 1, 3]),
        dict(shape=[1, 224, 1, 1, 100], axis=[0, 1, 2]),
        dict(shape=[1, 224, 1, 1, 1], axis=[0, 1, 2, 3])
    ]

    # TODO mark as precommit (after successfully passing in nightly)
    @pytest.mark.special_xfail(
        args={'ie_device': 'GPU', 'precision': 'FP16', 'params': {'axis': [0, 3]}},
        reason="*-19394")
    @pytest.mark.special_xfail(
        args={'ie_device': 'GPU', 'precision': 'FP16', 'params': {'axis': [0, 1, 3]}},
        reason="*-19394")
    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_squeeze_5D(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                        use_old_api):
        self._test(*self.create_squeeze_net(**params, ir_version=ir_version,
                                            use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
