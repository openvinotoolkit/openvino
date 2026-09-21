// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>

#include "openvino/core/node.hpp"
#include "openvino/core/type.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"

namespace ov::intel_gpu {

struct WeightedReduceMatch {
    std::shared_ptr<ov::op::v1::Multiply> multiply;
    size_t values_idx = 0;
    size_t weights_idx = 1;
};

/// @brief Returns the operands of an eligible Multiply followed by ReduceSum or ReduceMean.
inline std::optional<WeightedReduceMatch> get_weighted_reduce_match(const ov::Node* reduce_node) {
    if (reduce_node == nullptr || (!ov::is_type<ov::op::v1::ReduceSum>(reduce_node) && !ov::is_type<ov::op::v1::ReduceMean>(reduce_node))) {
        return std::nullopt;
    }

    const auto& reduce_input_pshape = reduce_node->get_input_partial_shape(0);
    if (reduce_input_pshape.rank().is_dynamic() || reduce_input_pshape.size() != 4) {
        return std::nullopt;
    }
    const auto rank = static_cast<int64_t>(reduce_input_pshape.size());

    const auto axes_constant = ov::as_type_ptr<ov::op::v0::Constant>(reduce_node->get_input_node_shared_ptr(1));
    if (!axes_constant) {
        return std::nullopt;
    }
    const auto axes = axes_constant->cast_vector<int64_t>();
    if (axes.size() != 1) {
        return std::nullopt;
    }
    const int64_t axis = axes[0] < 0 ? axes[0] + rank : axes[0];
    if (axis != rank - 1) {
        return std::nullopt;
    }

    auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(reduce_node->get_input_node_shared_ptr(0));
    if (!multiply || multiply->get_output_target_inputs(0).size() != 1 || !multiply->get_input_partial_shape(0).is_static() ||
        !multiply->get_input_partial_shape(1).is_static()) {
        return std::nullopt;
    }

    const auto input_type = multiply->get_input_element_type(0);
    if (input_type != multiply->get_input_element_type(1) || (input_type != ov::element::f16 && input_type != ov::element::f32)) {
        return std::nullopt;
    }

    const auto shape0 = multiply->get_input_shape(0);
    const auto shape1 = multiply->get_input_shape(1);
    if (shape0.size() != 4 || shape1.size() != 4 || shape0[0] != shape1[0] || shape0[2] != shape1[2] || shape0[2] <= 1024 || shape0[3] != shape1[3] ||
        shape0[3] != 16) {
        return std::nullopt;
    }

    WeightedReduceMatch match{multiply};
    if (shape0[1] == 1 && shape1[1] > 1) {
        match.values_idx = 1;
        match.weights_idx = 0;
    } else if (shape1[1] != 1 || shape0[1] <= 1) {
        return std::nullopt;
    }

    return match;
}

/// @brief Returns true when a Multiply is consumed by an eligible weighted reduction.
inline bool is_weighted_reduce_multiply(const ov::Node* multiply_node) {
    if (multiply_node == nullptr) {
        return false;
    }
    const auto consumers = multiply_node->get_output_target_inputs(0);
    if (consumers.size() != 1) {
        return false;
    }
    const auto match = get_weighted_reduce_match(consumers.begin()->get_node());
    return match.has_value() && match->multiply.get() == multiply_node;
}

}  // namespace ov::intel_gpu
