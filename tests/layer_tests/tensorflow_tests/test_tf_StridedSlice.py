# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestStridedSlice(CommonTFLayerTest):
    def create_strided_slice_net(self, input_shape, begin_value, end_value, strides_value, begin_mask, end_mask,
                                 ellipsis_mask,
                                 new_axis_mask, shrink_axis_mask):
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()

        with tf.compat.v1.Session() as sess:
            input = tf.compat.v1.placeholder(tf.float32, input_shape, 'Input')
            begin = tf.constant(begin_value, dtype=tf.int32)
            end = tf.constant(end_value, dtype=tf.int32)
            strides = tf.constant(strides_value, dtype=tf.int32)
            tf.raw_ops.StridedSlice(input=input, begin=begin, end=end, strides=strides, begin_mask=begin_mask,
                                    end_mask=end_mask, ellipsis_mask=ellipsis_mask, new_axis_mask=new_axis_mask,
                                    shrink_axis_mask=shrink_axis_mask)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_basic_data = [
        dict(input_shape=[2, 5, 4, 3], begin_value=[1, 0, 2, 0], end_value=[2, 5, 4, 2], strides_value=[1, 2, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[1, 5, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 5, 3], strides_value=[1, 2, 3, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=8, shrink_axis_mask=0),
        dict(input_shape=[3, 4, 5, 7], begin_value=[2, 0, 3], end_value=[3, 0, 6], strides_value=[1, 1, 1],
             begin_mask=6, end_mask=6, ellipsis_mask=2, new_axis_mask=0, shrink_axis_mask=1),
    ]

    @pytest.mark.parametrize('params', test_basic_data)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    def test_strided_slice_basic(self, params, ie_device, precision, ir_version,
                                 temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_strided_slice_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_squeeze_data = [
        dict(input_shape=[1, 5], begin_value=[0, 0], end_value=[1, 5], strides_value=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[5, 1], begin_value=[0, 0], end_value=[5, 1], strides_value=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=1),
        dict(input_shape=[1, 1, 3], begin_value=[0, 0, 0], end_value=[1, 1, 3], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 1], begin_value=[0, 0, 0], end_value=[1, 5, 1], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=4),
        dict(input_shape=[1, 1, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 1, 5, 3], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=2),
        dict(input_shape=[1, 5, 1, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 1, 3], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=4),
        dict(input_shape=[1, 5, 5, 1], begin_value=[0, 0, 0, 0], end_value=[1, 5, 1, 1], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=8),
        dict(input_shape=[1, 1, 5, 5, 3], begin_value=[0, 0, 0, 0, 0], end_value=[1, 1, 5, 5, 3],
             strides_value=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=3),
        dict(input_shape=[1, 5, 1, 5, 3], begin_value=[0, 0, 0, 0, 0], end_value=[1, 5, 1, 5, 3],
             strides_value=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=5),
        dict(input_shape=[1, 5, 1, 5, 1], begin_value=[0, 0, 0, 0, 0], end_value=[1, 5, 1, 5, 1],
             strides_value=[1, 1, 1, 1, 1],
             begin_mask=0, end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=21),
    ]

    @pytest.mark.parametrize('params', test_squeeze_data)
    @pytest.mark.nightly
    def test_strided_slice_replace_with_squeeze(self, params, ie_device, precision, ir_version,
                                                temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_strided_slice_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)

    test_unsqueeze_data = [
        dict(input_shape=[1, 5], begin_value=[0, 0], end_value=[1, 5], strides_value=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=1, shrink_axis_mask=0),
        dict(input_shape=[1, 5], begin_value=[0, 0], end_value=[1, 5], strides_value=[1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=3, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=3, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=4, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 3], begin_value=[0, 0, 0], end_value=[1, 5, 3], strides_value=[1, 1, 1], begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=5, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 5, 3], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=4, shrink_axis_mask=0),
        dict(input_shape=[1, 5, 5, 3], begin_value=[0, 0, 0, 0], end_value=[1, 5, 5, 3], strides_value=[1, 1, 1, 1],
             begin_mask=0,
             end_mask=0, ellipsis_mask=0, new_axis_mask=2, shrink_axis_mask=0),
        dict(input_shape=[16, 4, 64], begin_value=[0, 0, 0, 0], end_value=[0, 0, 0, 0], strides_value=[1, 1, 1, 1],
             begin_mask=19,
             end_mask=19, ellipsis_mask=0, new_axis_mask=12, shrink_axis_mask=0),
    ]

    @pytest.mark.parametrize('params', test_unsqueeze_data)
    @pytest.mark.nightly
    def test_strided_slice_replace_with_unsqueeze(self, params, ie_device, precision, ir_version,
                                                  temp_dir, use_new_frontend, use_old_api):
        self._test(*self.create_strided_slice_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_new_frontend=use_new_frontend, use_old_api=use_old_api)
