// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/label.hpp"

#include "openvino/core/log_util.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/true.hpp"
#include "openvino/util/log.hpp"

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

ov::Output<ov::Node> ov::pass::pattern::op::Label::wrap_values(const ov::NodeVector& wrapped_values) {
    switch (wrapped_values.size()) {
    case 0:
        return std::make_shared<pattern::op::True>()->output(0);
    case 1:
        return wrapped_values[0];
    default:
        return std::make_shared<pattern::op::Or>(as_output_vector(wrapped_values))->output(0);
    }
}

bool ov::pass::pattern::op::Label::match_value(ov::pass::pattern::Matcher* matcher,
                                               const ov::Output<ov::Node>& pattern_value,
                                               const ov::Output<ov::Node>& graph_value) {
    if (m_predicate(matcher->get_symbols(), graph_value)) {
        auto& pattern_map = matcher->get_pattern_value_map();
        auto saved = matcher->start_match();
        matcher->add_node(graph_value);
        if (pattern_map.count(shared_from_this())) {
            OPENVINO_LOG_LABEL1(matcher, get_name());
            return saved.finish(pattern_map[shared_from_this()] == graph_value);
        } else {
            pattern_map[shared_from_this()] = graph_value;
            OPENVINO_LOG_LABEL2(matcher, get_name());
            auto res = saved.finish(matcher->match_value(input_value(0), graph_value));
            OPENVINO_LOG_LABEL3(matcher);
            return res;
        }
    }
    OPENVINO_LOG_LABEL4(matcher);
    return false;
}

std::shared_ptr<ov::Node> ov::pass::pattern::any_input() {
    return std::make_shared<pattern::op::Label>();
}
