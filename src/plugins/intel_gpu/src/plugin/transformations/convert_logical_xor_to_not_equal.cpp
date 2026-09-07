// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_logical_xor_to_not_equal.hpp"

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/logical_xor.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ov::intel_gpu {

ConvertLogicalXorToNotEqual::ConvertLogicalXorToNotEqual() {
    const auto pattern = ov::pass::pattern::wrap_type<ov::op::v1::LogicalXor>();
    const ov::matcher_pass_callback callback = [this](ov::pass::pattern::Matcher& matcher) {
        const auto logical = ov::as_type_ptr<ov::op::v1::LogicalXor>(matcher.get_match_root());
        if (!logical || transformation_callback(logical)) {
            return false;
        }
        const auto left = logical->input_value(0);
        const auto right = logical->input_value(1);
        if (left.get_element_type() != ov::element::boolean || right.get_element_type() != ov::element::boolean) {
            return false;
        }
        // For Boolean inputs XOR is inequality. Numeric truth-value conversions
        // are deliberately outside this rewrite's contract.
        const auto comparison = std::make_shared<ov::op::v1::NotEqual>(left, right, logical->get_autob());
        comparison->set_friendly_name(logical->get_friendly_name());
        ov::copy_runtime_info(logical, comparison);
        ov::replace_node(logical, comparison);
        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ConvertLogicalXorToNotEqual"), callback);
}

}  // namespace ov::intel_gpu
