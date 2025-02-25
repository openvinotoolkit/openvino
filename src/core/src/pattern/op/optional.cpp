// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/optional.hpp"

#include "openvino/core/log_util.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/log.hpp"

using namespace ov::pass::pattern::op;

std::vector<ov::DiscreteTypeInfo> Optional::get_optional_types() const {
    return optional_types;
}

bool Optional::match_value(Matcher* matcher, const Output<Node>& pattern_value, const Output<Node>& graph_value) {
    // Turn the Optional node into WrapType node to create a case where the Optional node is present
    ov::OutputVector input_values_to_optional = input_values();
    bool is_empty_in_values = input_values_to_optional.empty();
    auto wrap_node = std::make_shared<WrapType>(optional_types, m_predicate, input_values_to_optional);

    // Either continue using the WrapType if there're no inputs to it or create an Or node,
    // if there're other inputs to Optional creating another "branch" for matching.
    // Use only the 0th input as a "data" input. (To be changed or considered when Optional
    // starts supporting multiple inputs)
    auto or_node = is_empty_in_values ? std::static_pointer_cast<Pattern>(wrap_node)
                                      : std::static_pointer_cast<Pattern>(std::make_shared<Or>(
                                            ov::OutputVector{wrap_node, input_values_to_optional[0]}));
    OPENVINO_LOG_OPTIONAL1(matcher, or_node, wrap_node, get_name());
    if (matcher->match_value(or_node, graph_value)) {
        auto& pattern_map = matcher->get_pattern_value_map();
        if (pattern_map.count(wrap_node)) {
            pattern_map[shared_from_this()] = graph_value;
        }
        OPENVINO_LOG_OPTIONAL2(matcher);
        return true;
    }
    OPENVINO_LOG_OPTIONAL3(matcher);
    return false;
}
