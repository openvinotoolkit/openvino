# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestStridedSlice(CommonTFLayerTest):

    @staticmethod
    def create_strided_slice_net(input_shape, begin, end, strides, begin_mask, end_mask,
                                 ellipsis_mask,
                                 new_axis_mask, shrink_axis_mask, ir_version, use_new_frontend):

        import tensorflow as tf

        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            input_node = tf.compat.v1.placeholder(tf.float32, input_shape, 'Input')
            strided_slice = tf.compat.v1.strided_slice(input_node, begin=begin, end=end,
                                                       strides=strides,
                                                       begin_mask=begin_mask, end_mask=end_mask,
                                                       ellipsis_mask=ellipsis_mask,
                                                       new_axis_mask=new_axis_mask,
                                                       shrink_axis_mask=shrink_axis_mask)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_squeeze_data = [
        dict(input_shape=[1, 5], begin=[0, 0], end=[1, 5], strides=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[5, 1], begin=[0, 0], end=[5, 1], strides=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 3], begin=[0, 0, 0], end=[1, 5, 3], strides=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[1, 1, 3], begin=[0, 0, 0], end=[1, 1, 3], strides=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 1], begin=[0, 0, 0], end=[1, 5, 1], strides=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=4),
        pytest.param(dict(input_shape=[1, 5, 5, 3], begin=[0, 0, 0, 0], end=[1, 5, 5, 3], strides=[1, 1, 1, 1],
                          begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
                     marks=pytest.mark.precommit_tf_fe),
        dict(input_shape=[1, 1, 5, 3], begin=[0, 0, 0, 0], end=[1, 1, 5, 3], strides=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 1, 3], begin=[0, 0, 0, 0], end=[1, 5, 1, 3], strides=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=4),
        dict(input_shape=[1, 5, 5, 1], begin=[0, 0, 0, 0], end=[1, 5, 1, 1], strides=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=8),
        dict(input_shape=[1, 1, 5, 5, 3], begin=[0, 0, 0, 0, 0], end=[1, 1, 5, 5, 3],
             strides=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=3),
        dict(input_shape=[1, 5, 1, 5, 3], begin=[0, 0, 0, 0, 0], end=[1, 5, 1, 5, 3],
             strides=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=5),
        dict(input_shape=[1, 5, 1, 5, 1], begin=[0, 0, 0, 0, 0], end=[1, 5, 1, 5, 1],
             strides=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=21),
    ]

    @pytest.mark.parametrize('params', test_squeeze_data)
    @pytest.mark.nightly
    def test_strided_slice_replace_with_squeeze(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_strided_slice_net(**params, ir_version=ir_version,
                                                  use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_unsqueeze_data = [
        dict(input_shape=[1, 5], begin=[0, 0], end=[1, 5], strides=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=1, shrink_axis_mask=0),
        dict(input_shape=[1, 5], begin=[0, 0], end=[1, 5], strides=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=3, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 3], begin=[0, 0, 0], end=[1, 5, 3], strides=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=3, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 3], begin=[0, 0, 0], end=[1, 5, 3], strides=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=4, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 3], begin=[0, 0, 0], end=[1, 5, 3], strides=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=5, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 5, 3], begin=[0, 0, 0, 0], end=[1, 5, 5, 3], strides=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=8, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 5, 3], begin=[0, 0, 0, 0], end=[1, 5, 5, 3], strides=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=4, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 5, 3], begin=[0, 0, 0, 0], end=[1, 5, 5, 3], strides=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=2, shrink_axis_mask=0),
        dict(input_shape=[16, 4, 64], begin=[0, 0, 0, 0], end=[0, 0, 0, 0], strides=[1, 1, 1, 1],
             begin_mask=19,
             end_mask=19, ellipsis_mask=0, new_axis_mask=12, shrink_axis_mask=0),
    ]

    @pytest.mark.parametrize('params', test_unsqueeze_data)
    @pytest.mark.nightly
    def test_strided_slice_replace_with_unsqueeze(self, params, ie_device, precision, ir_version,
                                                  temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_strided_slice_net(**params, ir_version=ir_version,
                                                  use_new_frontend=use_new_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
