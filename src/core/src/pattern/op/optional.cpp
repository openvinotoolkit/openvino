// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/pass/pattern/matcher.hpp"

bool ov::pass::pattern::op::Optional::match_value(
    Matcher* matcher,
    const Output<Node>& pattern_value,
    const Output<Node>& graph_value) {
    std::cout << "Pattern_value: " << pattern_value.get_node_shared_ptr()->get_type_info() << " " << pattern_value.get_node_shared_ptr()->get_friendly_name() << std::endl;
    std::cout << "Graph_value: " << graph_value.get_node_shared_ptr()->get_type_info() << " " << graph_value.get_node_shared_ptr()->get_friendly_name() << std::endl;
    for (auto input_value : input_values()) {
        std::cout << "In_value: " << input_value.get_node_shared_ptr()->get_type_info() << " " << input_value.get_node_shared_ptr()->get_friendly_name() << std::endl;
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value, graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[input_value.get_node_shared_ptr()] = graph_value;
            return saved.finish(true);
        } else {
            for (auto next_level_in_value : graph_value.get_node()->input_values()) {
                std::cout << "NEXT_In_value: " << next_level_in_value.get_node_shared_ptr()->get_type_info() << " " << next_level_in_value.get_node_shared_ptr()->get_friendly_name() << std::endl;
                auto saved = matcher->start_match();
                if (matcher->match_value(input_value, next_level_in_value)) {
                    auto& pattern_map = matcher->get_pattern_value_map();
                    pattern_map[input_value.get_node_shared_ptr()] = next_level_in_value;
                    return saved.finish(true);
                }
            }
        }
    }
    return false;
    // return true;
}
