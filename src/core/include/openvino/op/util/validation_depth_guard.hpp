// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {

constexpr size_t kMaxValidationDepth = 64;

/// RAII depth guard for subgraph validation recursion (CWE-674).
/// Increments a per-op-type thread_local counter on construction, checks
/// the limit, and decrements on destruction.
class ValidationDepthGuard {
public:
    ValidationDepthGuard(size_t& depth, const Node* node, const char* op_name) : m_depth(depth) {
        ++m_depth;
        NODE_VALIDATION_CHECK(node,
                              m_depth <= kMaxValidationDepth,
                              op_name,
                              " nesting depth exceeds the maximum allowed limit of ",
                              kMaxValidationDepth);
    }
    ~ValidationDepthGuard() {
        --m_depth;
    }
    ValidationDepthGuard(const ValidationDepthGuard&) = delete;
    ValidationDepthGuard& operator=(const ValidationDepthGuard&) = delete;

private:
    size_t& m_depth;
};

// Usage: OV_VALIDATION_DEPTH_GUARD(this, "If");
#define OV_VALIDATION_DEPTH_GUARD(node, op_name)                      \
    static thread_local size_t ov_validation_depth_ = 0;              \
    ov::op::util::ValidationDepthGuard ov_depth_guard_(ov_validation_depth_, node, op_name)

}  // namespace util
}  // namespace op
}  // namespace ov
