// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/pass/pattern/matcher.hpp"

std::vector<ov::DiscreteTypeInfo> ov::pass::pattern::op::Optional::get_optional_types() const {
    return optional_types;
}

bool ov::pass::pattern::op::Optional::match_value(Matcher* matcher,
                                                  const Output<Node>& pattern_value,
                                                  const Output<Node>& graph_value) {
    auto& pattern_map = matcher->get_pattern_value_map();
    auto saved = matcher->start_match();
    auto updated_pattern_value = input_value(0);

    // find first pattern matched node from the end to the last node of the graph excluding optional operations
    // return true in case matched nodes were found. Otherside false
    const auto find_matched_node = [&matcher, &pattern_map](Output<Node>& p_value,
                                                            const Output<Node>& g_value,
                                                            const std::vector<ov::DiscreteTypeInfo>& optional_types) {
        do {
            auto& matched_values = matcher->get_matched_values();
            if (matcher->match_value(p_value, g_value) ||
                std::find(matched_values.begin(), matched_values.end(), g_value) != matched_values.end()) {
                pattern_map[p_value.get_node_shared_ptr()] = g_value;
                return true;
            }

            const auto pattern_node = p_value.get_node_shared_ptr();
            // optional op types should contain only one input to be replaced without extra input nodes
            if (std::find(optional_types.begin(), optional_types.end(), pattern_node->get_type_info()) ==
                    optional_types.end() ||
                pattern_node->get_input_size() != 1) {
                return false;
            }
            p_value = pattern_node->input(0).get_source_output();
        } while (p_value.get_node_shared_ptr() != nullptr);
        return false;
    };

    if (!find_matched_node(updated_pattern_value, graph_value, optional_types)) {
        return false;
    }

    const auto& pattern_node = updated_pattern_value.get_node_shared_ptr();
    const auto& graph_node = graph_value.get_node_shared_ptr();
    size_t in_cnt = pattern_node->get_input_size();
    if (in_cnt != graph_node->get_input_size()) {
        return false;
    }
    for (size_t in_idx = 0; in_idx < in_cnt; ++in_idx) {
        auto in_pattern_value = pattern_node->input(in_idx).get_source_output();
        auto in_graph_value = graph_node->input(in_idx).get_source_output();
        if (!find_matched_node(in_pattern_value, in_graph_value, optional_types)) {
            return false;
        }
    }
    return saved.finish(true);
}
