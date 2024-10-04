// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mark_rope_input_to_keep_in_mixed_precision.hpp"

#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

ov::pass::MarkRopeInputsToKeepInMixedPrecision::MarkRopeInputsToKeepInMixedPrecision() {
    MATCHER_SCOPE(MarkRopeInputsToKeepInMixedPrecision);
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;
    auto cos_tab = any_input();
    auto sin_tab = any_input();
    auto rope = makePattern<ov::op::internal::RoPE>({any_input(), cos_tab, sin_tab});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto cos_input_node = pattern_map.at(cos_tab).get_node();
        auto sin_input_node = pattern_map.at(sin_tab).get_node();
        // mark the node as disable_fp16_compression
        auto visit_func = [](ov::Node* node) {
            ov::disable_fp16_compression(node->shared_from_this());
        };
        // skip constant and parameter node
        auto skip_node_predicate = [](ov::Node* node) -> bool {
            return ov::is_type<ov::op::v0::Constant>(node) || ov::is_type<ov::op::v0::Parameter>(node);
        };
        if (!visited.count(cos_input_node)) {
            ov::op::util::visit_path(cos_input_node, visited, visit_func, skip_node_predicate);
        }
        if (!visited.count(sin_input_node)) {
            ov::op::util::visit_path(sin_input_node, visited, visit_func, skip_node_predicate);
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, matcher_name);
    this->register_matcher(m, callback);
}