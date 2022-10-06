# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.layer_test_class import check_ir_version
from common.tf_layer_test_class import CommonTFLayerTest

from unit_tests.utils.graph import build_graph


class TestBucketize(CommonTFLayerTest):
    def create_bucketize_net(self, input_shape, input_type, boundaries_size, ir_version,
                             use_new_frontend):
        """
            Tensorflow net:                     IR net:
                 Input            =>      Input        Boundaries
                   |                           \       /
               Bucketize                       Bucketize
           {attrs: boundaries}
        """

        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            x = tf.compat.v1.placeholder(input_type, input_shape, 'Input')
            constant_value = np.arange(-boundaries_size * 5, boundaries_size * 5, 10,
                                       dtype=np.float32)
            # TODO: Bucketize is not tested here. Need to re-write the test
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        # create reference IR net
        ref_net = None

        if check_ir_version(10, None, ir_version) and not use_new_frontend:
            nodes_attributes = {
                'input': {'kind': 'op', 'type': 'Parameter'},
                'input_data': {'shape': input_shape, 'kind': 'data'},
                'boundaries_input_data': {'shape': constant_value.shape, 'kind': 'data'},
                'boundaries': {'type': 'Const', 'kind': 'op'},
                'boundaries_data': {'kind': 'data', 'shape': constant_value.shape},
                'bucketize': {'kind': 'op', 'type': 'Bucketize'},
                'bucketize_data': {'shape': input_shape, 'kind': 'data'},
                'result': {'kind': 'op', 'type': 'Result'}
            }

            ref_net = build_graph(nodes_attributes,
                                  [('input', 'input_data'),
                                   ('input_data', 'bucketize', {'in': 0}),
                                   ('boundaries_input_data', 'boundaries'),
                                   ('boundaries', 'boundaries_data'),
                                   ('boundaries_data', 'bucketize', {'in': 1}),
                                   ('bucketize', 'bucketize_data'),
                                   ('bucketize_data', 'result')
                                   ])

        return tf_net, ref_net

    test_data_float32 = [
        dict(input_shape=[5], input_type=tf.float32, boundaries_size=1),
        dict(input_shape=[5], input_type=tf.float32, boundaries_size=3),
        pytest.param(dict(input_shape=[4, 8], input_type=tf.float32, boundaries_size=5),
                     marks=pytest.mark.precommit_tf_fe),
        dict(input_shape=[2, 4, 7], input_type=tf.float32, boundaries_size=10),
        dict(input_shape=[2, 4, 7, 8], input_type=tf.float32, boundaries_size=12),
        dict(input_shape=[2, 4, 7, 8, 10], input_type=tf.float32, boundaries_size=14)]

    @pytest.mark.parametrize("params", test_data_float32)
    @pytest.mark.nightly
    def test_bucketize_float32(self, params, ie_device, precision, ir_version, temp_dir,
                               use_new_frontend, use_old_api):
        self._test(*self.create_bucketize_net(**params, ir_version=ir_version,
                                              use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_data_int32 = [
        dict(input_shape=[5], input_type=tf.int32, boundaries_size=1),
        dict(input_shape=[5], input_type=tf.int32, boundaries_size=3),
        dict(input_shape=[4, 8], input_type=tf.int32, boundaries_size=5),
        dict(input_shape=[2, 4, 7], input_type=tf.int32, boundaries_size=10),
        dict(input_shape=[2, 4, 7, 8], input_type=tf.float32, boundaries_size=12),
        dict(input_shape=[2, 4, 7, 8, 10], input_type=tf.float32, boundaries_size=14)]

    @pytest.mark.parametrize("params", test_data_int32)
    @pytest.mark.nightly
    def test_bucketize_int32(self, params, ie_device, precision, ir_version, temp_dir,
                             use_new_frontend, use_old_api):
        self._test(*self.create_bucketize_net(**params, ir_version=ir_version,
                                              use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
