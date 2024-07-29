// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "markup_rope_inputs.hpp"

#include <unordered_set>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

ov::intel_cpu::MarkUpRopeInputs::MarkUpRopeInputs() {
    MATCHER_SCOPE(MarkUpRopeInputs);
    using namespace ov::pass::pattern;
    using namespace ov::gen_pattern;
    auto cos_tab = any_input();
    auto sin_tab = any_input();
    auto rope = makePattern<ov::op::internal::RoPE>({any_input(), cos_tab, sin_tab});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto cos_input_node = pattern_map.at(cos_tab).get_node_shared_ptr();
        auto sin_input_node = pattern_map.at(sin_tab).get_node_shared_ptr();
        auto bfs_markup = [&](std::shared_ptr<ov::Node>& input) {
            std::deque<std::shared_ptr<ov::Node>> nodes;
            nodes.push_back(input);
            while (!nodes.empty()) {
                auto curr_node = nodes.front();
                nodes.pop_front();
                visited.insert(curr_node);
                // visit cur node
                ov::disable_fp16_compression(curr_node);
                // extend parent nodes
                for (auto& input_value : curr_node->input_values()) {
                    const auto& input_node = input_value.get_node_shared_ptr();
                    if (visited.count(input_node)) {
                        continue;
                    }
                    if (!ov::is_type<ov::op::v0::Constant>(input_node) && !ov::is_type<ov::op::v0::Parameter>(input_node))
                        nodes.push_front(input_node);
                }
            }
        };
        if (!visited.count(cos_input_node)) {
            bfs_markup(cos_input_node);
        }
        if (!visited.count(sin_input_node)) {
            bfs_markup(sin_input_node);
        }
        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(rope, matcher_name);
    this->register_matcher(m, callback);
}