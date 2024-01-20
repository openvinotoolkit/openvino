// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/pass/pattern/matcher.hpp"

std::unordered_set<ov::DiscreteTypeInfo> ov::pass::pattern::op::Optional::get_optional_types() const {
    return optional_types;
}

bool ov::pass::pattern::op::Optional::match_value(Matcher* matcher,
                                                  const Output<Node>& pattern_value,
                                                  const Output<Node>& graph_value) {
    const auto p_value = input_value(0);
    auto g_value = graph_value;
    auto saved = matcher->start_match();
    auto& pattern_map = matcher->get_pattern_value_map();

    auto p_node = p_value.get_node_shared_ptr();
    if (matcher->match_value(p_value, g_value)) {
        pattern_map[p_node] = g_value;
        return saved.finish(true);
    }
    auto g_node = g_value.get_node_shared_ptr();
    if (g_node->get_input_size() != 1) {
        return false;
    }

    auto g_type_info = g_node->get_type_info();

    if (!std::any_of(optional_types.begin(), optional_types.end(), [&](const NodeTypeInfo& type_info) {
            return g_type_info.is_castable(type_info);
        })) {
        return false;
    }

    pattern_map[g_node] = g_value;
    g_value = g_node->input_value(0);
    if (matcher->match_value(p_value, g_value)) {
        pattern_map[p_node] = g_value;
        return saved.finish(true);
    }
    return false;
}
