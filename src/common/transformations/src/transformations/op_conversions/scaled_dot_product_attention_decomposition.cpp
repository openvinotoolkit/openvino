// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/scaled_dot_product_attention_decomposition.hpp"

#include <memory>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

ov::pass::ScaledDotProductAttentionDecomposition::ScaledDotProductAttentionDecomposition() {
    MATCHER_SCOPE(ScaledDotProductAttentionDecomposition);
    auto pattern_node = ov::pass::pattern::wrap_type<ov::op::v12::ScaledDotProductAttention>();

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto& pattern_to_output = m.get_pattern_value_map();
        auto node = std::dynamic_pointer_cast<ov::op::v12::ScaledDotProductAttention>(pattern_to_output.at(pattern_node).get_node_shared_ptr());

        if (node == nullptr || transformation_callback(node)) {
            return false;
        }

        auto new_output_node = node->decompose()[0].get_node_shared_ptr();
        new_output_node->set_friendly_name(node->get_friendly_name());
        ov::copy_runtime_info(node, new_output_node);
        ov::replace_node(node, new_output_node);
        std::cerr << "[ DEBUG ] ScaledDotProductAttention was decomposed in compile_model\n";
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern_node, matcher_name);
    register_matcher(m, callback);
}
