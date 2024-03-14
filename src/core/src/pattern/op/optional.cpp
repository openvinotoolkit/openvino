// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

std::vector<ov::DiscreteTypeInfo> ov::pass::pattern::op::Optional::get_optional_types() const {
    return optional_types;
}

bool ov::pass::pattern::op::Optional::match_value(Matcher* matcher,
                                                  const Output<Node>& pattern_value,
                                                  const Output<Node>& graph_value) {
    // Turn the Optional node into WrapType node to create a case where the Optional node is present
    ov::OutputVector input_values_to_optional = input_values();
    size_t num_input_values_to_optional = input_values_to_optional.size();
    auto wrap_node = std::make_shared<ov::pass::pattern::op::WrapType>(optional_types, m_predicate, input_values_to_optional);

    // Add the newly created WrapType node to the list containing its inputs and create an Or node with the list
    input_values_to_optional.push_back(wrap_node);
    auto or_node = std::make_shared<ov::pass::pattern::op::Or>(input_values_to_optional);

    if (matcher->match_value(or_node, graph_value) || num_input_values_to_optional == 0) {
        auto& pattern_map = matcher->get_pattern_value_map();
        if (pattern_map.count(wrap_node)) {
            pattern_map[shared_from_this()] = graph_value;
        }
        return true;
    }
    return false;
}
