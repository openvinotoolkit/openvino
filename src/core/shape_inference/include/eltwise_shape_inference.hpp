// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/util/binary_elementwise_comparison.hpp"
#include "openvino/op/util/binary_elementwise_logical.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
template <class OpType, class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> eltwise_shape_infer(const OpType* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2, "Incorrect number of input/output shapes");

    auto output_shapes = std::vector<TRShape>{input_shapes[0]};
    auto& output_shape = output_shapes[0];
    const auto& autob = op->get_autob();
    if (autob.m_type == AutoBroadcastType::NONE) {
        NODE_VALIDATION_CHECK(op,
                              TRShape::merge_into(output_shape, input_shapes[1]),
                              "Argument shapes are inconsistent.");
    } else if (autob.m_type == AutoBroadcastType::NUMPY || autob.m_type == AutoBroadcastType::PDPD) {
        NODE_VALIDATION_CHECK(op,
                              TRShape::broadcast_merge_into(output_shape, input_shapes[1], autob),
                              "Argument shapes are inconsistent.");
    } else {
        NODE_VALIDATION_CHECK(op, false, "Unsupported auto broadcast specification");
    }
    return output_shapes;
}
}  // namespace op
}  // namespace ov
