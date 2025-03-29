# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from common.tf_layer_test_class import CommonTFLayerTest


class TestConcat(CommonTFLayerTest):
    def create_concat_net(self, input_shapes, axis, is_v2, ir_version, use_legacy_frontend):
        # tf.concat is equivalent to tf.raw_ops.ConcatV2
        # only tf.concat accepts one input
        import tensorflow as tf
        tf.compat.v1.reset_default_graph()
        with tf.compat.v1.Session() as sess:
            placeholders = []
            for ind, input_shape in enumerate(input_shapes):
                placeholders.append(tf.compat.v1.placeholder(tf.float32, input_shape, 'input_{}'.format(ind)))
            if len(input_shapes) == 1:
                tf.concat(values=placeholders, axis=axis, name='concat')
            elif is_v2:
                tf.raw_ops.ConcatV2(values=placeholders, axis=axis, name='concat')
            else:
                tf.raw_ops.Concat(values=placeholders, concat_dim=axis, name='concat')
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        ref_net = None
        return tf_net, ref_net

    test_data_basic = [
        dict(input_shapes=[[1], [3]], axis=-1, is_v2=True),
        dict(input_shapes=[[3, 4]], axis=0, is_v2=True),
        dict(input_shapes=[[1, 3, 5], [2, 3, 5]], axis=0, is_v2=False),
        dict(input_shapes=[[1, 3, 5, 7, 8], [1, 3, 2, 7, 8]], axis=2, is_v2=False),
        dict(input_shapes=[[1, 3, 5, 7, 2], [1, 3, 5, 7, 3], [1, 3, 5, 7, 2], [1, 3, 5, 7, 4]],
             axis=-1, is_v2=True),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_concat_basic(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_1D = [
        dict(input_shapes=[[1], [2]], axis=0, is_v2=False),
    ]

    @pytest.mark.parametrize("params", test_data_1D)
    @pytest.mark.nightly
    def test_concat_1D(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_2D = [
        dict(input_shapes=[[1, 4], [1, 2]], axis=-1, is_v2=True)
    ]

    @pytest.mark.parametrize("params", test_data_2D)
    @pytest.mark.nightly
    def test_concat_2D(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_3D = [
        dict(input_shapes=[[1, 3, 2], [1, 3, 5]], axis=-1, is_v2=True),
        dict(input_shapes=[[1, 3, 1], [1, 3, 4], [1, 3, 3]], axis=2, is_v2=True)
    ]

    @pytest.mark.parametrize("params", test_data_3D)
    @pytest.mark.nightly
    def test_concat_3D(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_4D = [
        dict(input_shapes=[[1, 3, 5, 7], [3, 3, 5, 7], [2, 3, 5, 7]], axis=0, is_v2=False),
        dict(input_shapes=[[1, 3, 5, 5], [1, 3, 5, 7]], axis=-1, is_v2=True),
        dict(input_shapes=[[1, 3, 5, 7], [1, 3, 3, 7]], axis=2, is_v2=False)
    ]

    @pytest.mark.parametrize("params", test_data_4D)
    @pytest.mark.nightly
    def test_concat_4D(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)

    test_data_5D = [
        dict(input_shapes=[[1, 3, 5, 7, 8], [2, 3, 5, 7, 8]], axis=0, is_v2=True),
    ]

    @pytest.mark.parametrize("params", test_data_5D)
    @pytest.mark.nightly
    def test_concat_5D(self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend):
        self._test(*self.create_concat_net(**params, ir_version=ir_version,
                                           use_legacy_frontend=use_legacy_frontend),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)
