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
 
    auto in_value = input_value(0);

    std::list<Output<Node>> pattern_queue{in_value};
    while (!pattern_queue.empty()) {
        auto p_value = pattern_queue.front();
        pattern_queue.pop_front();
        auto saved = matcher->start_match();
        if (matcher->match_value(p_value, graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            if (!pattern_map.count(in_value.get_node_shared_ptr()))
                pattern_map[in_value.get_node_shared_ptr()] = graph_value;
            return saved.finish(true);
        }
        
        const auto p_node = p_value.get_node_shared_ptr();
        const auto p_type_info = p_node->get_type_info(); 
        if (!optional_types.count(p_type_info) ||
            p_node->get_input_size() != 1) {
            return false;
        }
        
        for (size_t in_idx = 0; in_idx < p_node->get_input_size(); ++in_idx) {
            pattern_queue.push_back(p_node->input_value(in_idx));
        }
    }
    return false;
}
