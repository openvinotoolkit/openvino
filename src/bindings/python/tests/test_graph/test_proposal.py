# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import openvino.opset8 as ov
from openvino import Shape, Type


def test_proposal_props():
    float_dtype = np.float32
    batch_size = 1
    post_nms_topn = 20
    probs = ov.parameter(Shape([batch_size, 8, 255, 255]), dtype=float_dtype, name="probs")
    deltas = ov.parameter(Shape([batch_size, 16, 255, 255]), dtype=float_dtype, name="bbox_deltas")
    im_info = ov.parameter(Shape([4]), dtype=float_dtype, name="im_info")

    attrs = {
        "base_size": np.uint32(85),
        "pre_nms_topn": np.uint32(10),
        "post_nms_topn": np.uint32(post_nms_topn),
        "nms_thresh": np.float32(0.34),
        "feat_stride": np.uint32(16),
        "min_size": np.uint32(32),
        "ratio": np.array([0.1, 1.5, 2.0, 2.5], dtype=np.float32),
        "scale": np.array([2, 3, 3, 4], dtype=np.float32),
    }

    node = ov.proposal(probs, deltas, im_info, attrs)

    assert node.get_type_name() == "Proposal"
    assert node.get_output_size() == 2

    assert list(node.get_output_shape(0)) == [batch_size * post_nms_topn, 5]
    assert list(node.get_output_shape(1)) == [batch_size * post_nms_topn]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_element_type(1) == Type.f32
