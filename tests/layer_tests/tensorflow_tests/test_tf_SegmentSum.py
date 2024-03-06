# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import platform

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest


class TestSegmentSum(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert "data" in inputs_info, "Test error: inputs_info must contain `data`"
        assert (
            "segment_ids" in inputs_info
        ), "Test error: inputs_info must contain `segment_ids`"
        data_shape = inputs_info["data"]
        segment_ids_shape = inputs_info["segment_ids"]
        inputs_data = {}
        inputs_data["data"] = np.random.randint(-50, 50, data_shape)
        # segment_ids data must be sorted according to TensorFlow SegmentSum specification
        inputs_data["segment_ids"] = np.sort(
            np.random.randint(0, 20, segment_ids_shape)
        )
        return inputs_data

    def create_segment_sum_net(
        self, data, segment_ids, data_type=tf.float32, segment_ids_type=tf.int32
    ):
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            data = tf.compat.v1.placeholder(data_type, data, "data")
            segment_ids = tf.compat.v1.placeholder(
                segment_ids_type, segment_ids, "segment_ids"
            )
            tf.math.segment_sum(data, segment_ids)
            tf.compat.v1.global_variables_initializer()

            tf_net = sess.graph_def

        return tf_net, None

    test_data_basic = [
        dict(
            data=[8],
            segment_ids=[8],
        ),
        dict(
            data=[3, 7],
            segment_ids=[3],
        ),
        dict(
            data=[4, 3, 2],
            segment_ids=[4],
        ),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit_tf_fe
    @pytest.mark.nightly
    @pytest.mark.xfail(
        condition=platform.system() == "Darwin" and platform.machine() == "arm64",
        reason="Ticket - 122716",
    )
    def test_segment_sum_basic(
        self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend
    ):
        self._test(
            *self.create_segment_sum_net(**params),
            ie_device,
            precision,
            ir_version,
            temp_dir=temp_dir,
            use_legacy_frontend=use_legacy_frontend
        )

    test_data_different_types = [
        dict(
            data=[2, 3],
            segment_ids=[2],
            data_type=tf.int32,
            segment_ids_type=tf.int32,
        ),
        dict(
            data=[3, 2],
            segment_ids=[3],
            data_type=tf.float64,
            segment_ids_type=tf.int32,
        ),
        dict(
            data=[3, 1, 2],
            segment_ids=[3],
            data_type=tf.float32,
            segment_ids_type=tf.int64,
        ),
        dict(
            data=[4, 2, 1],
            segment_ids=[4],
            data_type=tf.float64,
            segment_ids_type=tf.int64,
        ),
    ]

    @pytest.mark.parametrize("params", test_data_different_types)
    @pytest.mark.nightly
    def test_segment_sum_different_types(
        self, params, ie_device, precision, ir_version, temp_dir, use_legacy_frontend
    ):
        self._test(
            *self.create_segment_sum_net(**params),
            ie_device,
            precision,
            ir_version,
            temp_dir=temp_dir,
            use_legacy_frontend=use_legacy_frontend
        )
