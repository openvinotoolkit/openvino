// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "preserve_boolean_attn_mask_precision.hpp"

#include <unordered_set>
#include <vector>

#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {

PreserveBooleanAttnMaskPrecision::PreserveBooleanAttnMaskPrecision() {
    auto sdpa_pattern = ov::pass::pattern::wrap_type<ov::op::v13::ScaledDotProductAttention>();
    ov::matcher_pass_callback callback = [](ov::pass::pattern::Matcher& matcher) {
        auto sdpa = ov::as_type_ptr<ov::op::v13::ScaledDotProductAttention>(matcher.get_match_root());
        if (!sdpa->get_causal() && sdpa->get_input_size() >= 4 &&
            sdpa->get_input_element_type(3) == ov::element::boolean) {
            std::vector<ov::Output<ov::Node>> pending{sdpa->input_value(3)};
            std::unordered_set<ov::Node*> visited;
            while (!pending.empty()) {
                auto output = pending.back();
                pending.pop_back();

                auto producer = output.get_node_shared_ptr();
                if (!visited.insert(producer.get()).second) {
                    continue;
                }

                ov::disable_conversion(producer, ov::element::boolean, ov::element::u8);
                for (const auto& input : producer->inputs()) {
                    if (input.get_element_type() == ov::element::boolean) {
                        pending.push_back(input.get_source_output());
                    }
                }
            }
        }
        return false;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(sdpa_pattern, "PreserveBooleanAttnMaskPrecision"),
        callback);
}

}  // namespace ov::intel_gpu