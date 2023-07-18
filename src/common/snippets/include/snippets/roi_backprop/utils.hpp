// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/core/validation_util.hpp>
#include <openvino/opsets/opset1.hpp>

template <class OpType, class ShapeType>
void transparent_roi_backprop(const OpType* op,
                              const std::vector<ShapeType>& input_shapes,
                              const std::vector<ShapeType>& cur_roi,
                              const std::vector<ov::Shape>& cur_strides,
                              std::vector<ShapeType>& new_roi,
                              std::vector<ov::Shape>& new_strides) {
    NODE_VALIDATION_CHECK(op, cur_roi.size() == 1, "Incorrect number of current roi shapes");
    for (auto& roi_shape : new_roi)
        roi_shape = cur_roi[0];

    new_strides = cur_strides;
}
