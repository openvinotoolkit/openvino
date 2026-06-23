// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
namespace npuw {
namespace moe_utils {

// Returns true if the node's output is entirely determined by constants
// (no data-dependent values from Parameters flow through it).
// Recognizes: Constant, Convert(constant), Multiply(constant, constant).
inline bool is_constant_derived(const std::shared_ptr<ov::Node>& n) {
    if (!n)
        return false;
    if (std::dynamic_pointer_cast<ov::op::v0::Constant>(n))
        return true;
    if (auto conv = std::dynamic_pointer_cast<ov::op::v0::Convert>(n)) {
        return is_constant_derived(conv->input_value(0).get_node_shared_ptr());
    }
    if (auto mul = std::dynamic_pointer_cast<ov::op::v1::Multiply>(n)) {
        return is_constant_derived(mul->input_value(0).get_node_shared_ptr()) &&
               is_constant_derived(mul->input_value(1).get_node_shared_ptr());
    }
    return false;
}

}  // namespace moe_utils
}  // namespace npuw
}  // namespace ov
