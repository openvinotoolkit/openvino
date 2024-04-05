// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::pass::pattern::op;

std::vector<ov::DiscreteTypeInfo> ov::pass::pattern::op::Optional::get_optional_types() const {
    return optional_types;
}

bool ov::pass::pattern::op::Optional::match_value(Matcher* matcher,
                                                  const Output<Node>& pattern_value,
                                                  const Output<Node>& graph_value) {
    // Turn the Optional node into WrapType node to create a case where the Optional node is present
    ov::OutputVector input_values_to_optional = input_values();
    bool in_values_cnt = input_values_to_optional.size();
    auto wrap_node = std::make_shared<WrapType>(optional_types, m_predicate, input_values_to_optional);
    ov::OutputVector or_input_values{
        input_values_to_optional.front(),
        wrap_node,
    };
    // matching arguments in case of `optional`: input values size can be different
    // in this case pattern should cover all possible cases by `optional` pattern
    // against graph with `lost` inputs
    // Operation example: ov::op::v5::NMS
    {
        int in_idx = in_values_cnt;
        while (--in_idx > 0) {
            const auto pattern_in_node_type = input_values_to_optional[in_idx].get_node()->get_type_info();
            if (!pattern_in_node_type.is_castable(get_type_info_static())) {
                break;
            }
            auto it_begin = input_values_to_optional.begin();
            auto it_end = it_begin;
            std::advance(it_end, in_idx);
            ov::OutputVector wrap_type_inputs(it_begin, it_end);
            or_input_values.push_back(std::make_shared<WrapType>(optional_types, m_predicate, wrap_type_inputs));
        }
    }

    // Either continue using the WrapType if there're no inputs to it or create an Or node,
    // if there're other inputs to Optional creating another "branch" for matching.
    // Use only the 0th input as a "data" input. (To be changed or considered when Optional
    // starts supporting multiple inputs)
    auto pattern = in_values_cnt ? std::static_pointer_cast<Pattern>(wrap_node)
                                 : std::static_pointer_cast<Pattern>(std::make_shared<Or>(or_input_values));
    if (matcher->match_value(pattern, graph_value)) {
        auto& pattern_map = matcher->get_pattern_value_map();
        if (pattern_map.count(wrap_node)) {
            pattern_map[shared_from_this()] = graph_value;
        }
        return true;
    }
    return false;
}
