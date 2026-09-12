// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace npuw {
namespace moe_utils {

// Returns true if the node's output is entirely determined by constants
// (no data-dependent values from Parameters flow through it).
// Recognizes decompression arithmetic and group-quantized weight views without
// folding or materializing the (potentially very large) floating-point weights.
inline bool is_constant_derived(const std::shared_ptr<ov::Node>& n) {
    if (!n)
        return false;
    if (std::dynamic_pointer_cast<ov::op::v0::Constant>(n))
        return true;
    if (auto conv = std::dynamic_pointer_cast<ov::op::v0::Convert>(n)) {
        return is_constant_derived(conv->input_value(0).get_node_shared_ptr());
    }
    if (ov::is_type<ov::op::v1::Multiply>(n) || ov::is_type<ov::op::v1::Subtract>(n) ||
        ov::is_type<ov::op::v1::Add>(n) || ov::is_type<ov::op::v1::Reshape>(n)) {
        return is_constant_derived(n->input_value(0).get_node_shared_ptr()) &&
               is_constant_derived(n->input_value(1).get_node_shared_ptr());
    }
    return false;
}

}  // namespace moe_utils
}  // namespace npuw
}  // namespace ov
