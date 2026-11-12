// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::op::internal::qsa_util {

inline void check_i32_input(const ov::Node* node, size_t idx, const char* name, int64_t rank) {
    const auto& et = node->get_input_element_type(idx);
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node,
                          et.is_dynamic() || et == element::i32,
                          node->get_type_name(),
                          " expects an i32 ",
                          name,
                          " input, got ",
                          et);
    NODE_VALIDATION_CHECK(node,
                          ps.rank().compatible(rank),
                          node->get_type_name(),
                          " expects a rank-",
                          rank,
                          " ",
                          name,
                          " input, got ",
                          ps);
}

inline void check_shape(const ov::Node* node, size_t idx, const char* name, const ov::PartialShape& expected) {
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node,
                          ps.compatible(expected),
                          node->get_type_name(),
                          " expects ",
                          name,
                          " shaped ",
                          expected,
                          ", got ",
                          ps);
}

inline ov::Dimension dim_at(const ov::PartialShape& ps, size_t axis) {
    return ps.rank().is_static() && ps.size() > axis ? ps[axis] : ov::Dimension::dynamic();
}

}  // namespace ov::op::internal::qsa_util
