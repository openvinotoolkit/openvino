// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

std::unordered_set<ov::DiscreteTypeInfo> ov::pass::pattern::op::Optional::get_optional_types() const {
    return optional_types;
}

bool ov::pass::pattern::op::Optional::match_value(Matcher* matcher,
                                                  const Output<Node>& pattern_value,
                                                  const Output<Node>& graph_value) {
    const auto p_value = input_value(0);
    const auto p_node = p_value.get_node_shared_ptr();
    auto& pattern_map = matcher->get_pattern_value_map();
    if (matcher->match_value(p_value, graph_value)) {
        pattern_map[p_node] = graph_value;
        return true;
    }

    auto wrap_type_op = std::dynamic_pointer_cast<ov::pass::pattern::op::WrapType>(p_node);
    if (wrap_type_op != nullptr) {
        for (const auto& wrapped_type : wrap_type_op->get_wrapped_types()) {
            if (!optional_types.count(wrapped_type)) {
                return false;
            }
        }
    } else if (!std::any_of(optional_types.begin(),
                            optional_types.end(),
                            [&](const NodeTypeInfo& type_info) {
                                return p_node->get_type_info().is_castable(type_info);
                            }) &&
               m_predicate(graph_value)) {
        return false;
    }

    for (const auto& input_value : p_node->input_values()) {
        auto in_node = input_value.get_node_shared_ptr();
        if (matcher->match_value(in_node, graph_value)) {
            pattern_map[in_node] = graph_value;
            return true;
        }
    }
    return false;
}
