# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestGather(CommonTFLayerTest):

    def create_indices_constant(self):
        pass

    def create_gather_net(self, data_shape, indices, axis, batch_dims, use_new_frontend, **kwargs):
        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(tf.float32, data_shape, 'data')
            indices = tf.constant(indices, dtype=tf.int32)
            gather = tf.gather(data, indices, axis=axis, batch_dims=batch_dims,
                               name='gather_output')

            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None

        return tf_net, ref_net

    test_data_precommit = [
        dict(data_shape=[6, 8, 10, 12], indices=[[0, 2, 4], [5, 7, 9]], axis=2, batch_dims=0),
        dict(data_shape=[4, 6, 8, 10, 12], indices=[2, 5], axis=1, batch_dims=0),
        dict(data_shape=[4, 6, 8, 10, 12], indices=[2, 5], axis=-1, batch_dims=0)
    ]

    @pytest.mark.parametrize("params", test_data_precommit)
    @pytest.mark.precommit
    def test_gather(self, params, ie_device, precision, ir_version, temp_dir, use_new_frontend,
                    use_old_api):
        self._test(*self.create_gather_net(**params, ir_version=ir_version,
                                           use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_nightly = [
        dict(data_shape=[2, 3], axis=1, indices=[0, 2], batch_dims=0),
        dict(data_shape=[10, 12], axis=0, indices=[3, 6], batch_dims=0),
        dict(data_shape=[10, 12], axis=1, indices=[[0, 1, 3, 4, 5], [6, 7, 9, 10, 11]],
             batch_dims=0),
        dict(data_shape=[8, 10, 12], axis=0, indices=[3, 6], batch_dims=0),
        pytest.param(dict(data_shape=[8, 10, 12], axis=-1, indices=[5, 8], batch_dims=0),
                     marks=pytest.mark.precommit_tf_fe),
        dict(data_shape=[6, 8, 10, 12], axis=0, indices=[2, 5], batch_dims=0),
        dict(data_shape=[6, 8, 10, 12], axis=-1, indices=[5, 8], batch_dims=0),
        dict(data_shape=[6, 8, 10, 12], axis=2, indices=[[0, 2, 4], [5, 7, 9]], batch_dims=0),
        dict(data_shape=[2, 14, 10, 12], axis=1, indices=[[0, 1, 3, 4, 5], [6, 7, 9, 10, 11]],
             batch_dims=1),
        dict(data_shape=[4, 6, 8, 10, 12], axis=0, indices=[1, 3], batch_dims=0),
        dict(data_shape=[4, 6, 8, 10, 12], axis=-1, indices=[5, 8], batch_dims=0),
    ]

    @pytest.mark.parametrize("params", test_data_nightly)
    @pytest.mark.nightly
    def test_gather_nightly(self, params, ie_device, precision, ir_version, temp_dir,
                            use_new_frontend, use_old_api):
        self._test(*self.create_gather_net(**params, use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
