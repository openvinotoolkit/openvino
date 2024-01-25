// Copyright (C) 2018-2023 Intel Corporation
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
    ov::OutputVector or_in_values{input_value(0)};
    or_in_values.push_back(
        std::make_shared<ov::pass::pattern::op::WrapType>(optional_types, m_predicate, or_in_values));

    if (matcher->match_value(std::make_shared<ov::pass::pattern::op::Or>(or_in_values), graph_value)) {
        auto& pattern_map = matcher->get_pattern_value_map();
        pattern_map[shared_from_this()] = graph_value;
        return true;
    }
    return false;
}
