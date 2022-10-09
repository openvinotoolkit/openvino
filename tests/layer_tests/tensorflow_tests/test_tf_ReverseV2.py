# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
from common.tf_layer_test_class import CommonTFLayerTest


class TestReverseV2Ops(CommonTFLayerTest):
    def _prepare_input(self, inputs_dict):
        for input in inputs_dict.keys():
            inputs_dict[input] = np.random.random(inputs_dict[input])
        return inputs_dict

    def create_reversev2_net(self, shape, keep_dims, axis, ir_version):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            shapes = shape.copy()
            if len(shapes) >= 4:
                shapes.append(shapes.pop(1))

            x = tf.compat.v1.placeholder(tf.float32, shapes, 'Input')
            tf.compat.v1.reverse_v2(x, axis)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    test_data = []
    test_data.extend([
        dict(shape=[5], axis=[0]),
        pytest.param(dict(shape=[2, 3], axis=[1]), marks=pytest.mark.precommit_tf_fe),
        dict(shape=[2, 3, 5], axis=[-2]),
        dict(shape=[2, 3, 5, 7], axis=[0]),
    ])

    @pytest.mark.parametrize("params", test_data)
    @pytest.mark.parametrize("keep_dims", [True, False])
    @pytest.mark.nightly
    def test_reversev2(self, params, keep_dims, ie_device, precision, ir_version, temp_dir, use_old_api):
        self._test(*self.create_reversev2_net(**params, keep_dims=keep_dims, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir, use_old_api=use_old_api)

    test_data_pre_commit = []
    test_data_pre_commit.extend([dict(shape=[5], axis=[0]),
                                 dict(shape=[2, 3, 5], axis=[-2])
                                 ])

    @pytest.mark.parametrize("params", test_data_pre_commit)
    @pytest.mark.parametrize("keep_dims", [True])
    @pytest.mark.precommit
    def test_reversev2_precommit(self, params, keep_dims, ie_device, precision, ir_version,
                                 temp_dir, use_old_api):
        self._test(*self.create_reversev2_net(**params, keep_dims=keep_dims, ir_version=ir_version),
                   ie_device, precision, ir_version, temp_dir=temp_dir, use_old_api=use_old_api, use_new_frontend=False)
