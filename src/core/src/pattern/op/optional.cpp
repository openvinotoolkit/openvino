// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/pass/pattern/matcher.hpp"

bool ov::pass::pattern::op::Optional::match_value(
    Matcher* matcher,
    const Output<Node>& pattern_value,
    const Output<Node>& graph_value) {
    auto in_value = input_value(0);
    auto& pattern_map = matcher->get_pattern_value_map();

    auto match_value = [&pattern_map, &matcher](const Output<Node>& p_value, const Output<Node>& g_value) {
        if (matcher->match_value(p_value, g_value)) {
            pattern_map[p_value.get_node_shared_ptr()] = g_value;
            return true;
        }
        return false;
    };

    auto saved = matcher->start_match();
    if (match_value(in_value, graph_value)) {
        return saved.finish(true);
    }
    const auto matched_values = matcher->get_matched_values();
    if (matched_values.empty() ||
        std::find(matched_values.begin(), matched_values.end(), graph_value) == matched_values.end()) {
        return false;
    }
    const auto pattern_node = in_value.get_node_shared_ptr();
    const auto graph_node = graph_value.get_node_shared_ptr();
    pattern_map[pattern_node] = graph_value;
    for (size_t in_idx = 0; in_idx < pattern_node->get_input_size(); ++in_idx) {
        in_value = pattern_node->input(in_idx).get_source_output();
        const auto graph_in_value = graph_node->input(in_idx).get_source_output();
        if (match_value(in_value, graph_in_value)) {
            continue;
        }
        const auto secondary_pattern_node = in_value.get_node_shared_ptr();
        if (std::find(optional_type.begin(), optional_type.end(), secondary_pattern_node->get_type_info()) != optional_type.end()) {
            if (match_value(matcher, in_value, graph_in_value)) {
                pattern_map[in_value.get_node_shared_ptr()] = graph_in_value;
            }
        }
        return false;
    }
    return saved.finish(true);
}
