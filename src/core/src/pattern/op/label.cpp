// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/label.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/true.hpp"

ov::Output<ov::Node> ov::pass::pattern::op::Label::wrap_values(const ov::OutputVector& wrapped_values) {
    switch (wrapped_values.size()) {
    case 0:
        return std::make_shared<pattern::op::True>()->output(0);
    case 1:
        return wrapped_values[0];
    default:
        return std::make_shared<pattern::op::Or>(wrapped_values)->output(0);
    }
}

bool ov::pass::pattern::op::Label::match_value(ov::pass::pattern::Matcher* matcher,
                                               const ov::Output<ov::Node>& pattern_value,
                                               const ov::Output<ov::Node>& graph_value) {
    if (m_predicate(graph_value)) {
        auto& pattern_map = matcher->get_pattern_value_map();
        auto saved = matcher->start_match();
        matcher->add_node(graph_value);
        if (pattern_map.count(shared_from_this())) {
            return saved.finish(pattern_map[shared_from_this()] == graph_value);
        } else {
            pattern_map[shared_from_this()] = graph_value;
            return saved.finish(matcher->match_value(input_value(0), graph_value));
        }
    }
    return false;
}

std::shared_ptr<ov::Node> ov::pass::pattern::any_input() {
    return std::make_shared<pattern::op::Label>();
}

std::shared_ptr<ov::Node> ov::pass::pattern::any_input(const ov::pass::pattern::op::ValuePredicate& pred) {
    return std::make_shared<pattern::op::Label>(element::dynamic, PartialShape::dynamic(), pred);
}
