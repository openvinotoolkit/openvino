// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>
#include <sstream>

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
        if (!TRShape::merge_into(output_shape, input_shapes[1])) {
            std::stringstream ss;
            ss << "[SD3-DBG] eltwise_shape_infer FAIL (NONE) op_type=" << op->get_type_name()
               << " name=" << op->get_friendly_name()
               << " in0=" << input_shapes[0] << " in1=" << input_shapes[1] << std::endl;
            std::cout << ss.str() << std::flush;
            std::cerr << ss.str() << std::flush;
        }
        NODE_VALIDATION_CHECK(op,
                              TRShape::merge_into(output_shape, input_shapes[1]),
                              "Argument shapes are inconsistent.");
    } else if (autob.m_type == AutoBroadcastType::NUMPY || autob.m_type == AutoBroadcastType::PDPD) {
        // Make a copy because broadcast_merge_into mutates output_shape on partial success.
        TRShape probe = output_shape;
        if (!TRShape::broadcast_merge_into(probe, input_shapes[1], autob)) {
            std::stringstream ss;
            ss << "[SD3-DBG] eltwise_shape_infer FAIL (broadcast=" << autob.m_type << ") op_type=" << op->get_type_name()
               << " name=" << op->get_friendly_name()
               << " in0=" << input_shapes[0] << " in1=" << input_shapes[1] << std::endl;
            std::cout << ss.str() << std::flush;
            std::cerr << ss.str() << std::flush;
        }
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
