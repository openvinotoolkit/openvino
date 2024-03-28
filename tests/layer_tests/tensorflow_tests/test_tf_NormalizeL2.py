# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestNormalizeL2(CommonTFLayerTest):
    @staticmethod
    def create_normalize_l2_net(shape, axes):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(tf.float32, shape=shape, name='data')
            tf.math.l2_normalize(data,
                                 axes,
                                 name='Operation')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(shape=[2, 3], axes=[1]),
        dict(shape=[2, 3, 5], axes=[1, -1]),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.precommit
    @pytest.mark.nightly
    @pytest.mark.xfail(condition=platform.system() in ('Linux', 'Darwin') and platform.machine() in ('arm', 'armv7l',
                                                                                                     'aarch64',
                                                                                                     'arm64', 'ARM64'),
                       reason='Ticket - 126314, 122716')
    def test_normalize_l2_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                use_legacy_frontend):
        self._test(*self.create_normalize_l2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_complex = [
        dict(shape=[2, 3, 5, 4], axes=[1, 2, 3]),
        dict(shape=[2, 3, 5, 4, 2], axes=[-1]),
    ]

    @pytest.mark.parametrize("params", test_data_complex)
    @pytest.mark.nightly
    def test_normalize_l2_complex(self, params, ie_device, precision, ir_version, temp_dir,
                                  use_legacy_frontend):
        self._test(*self.create_normalize_l2_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
