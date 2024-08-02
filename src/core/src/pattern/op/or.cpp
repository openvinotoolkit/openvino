// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/or.hpp"

#include "openvino/pass/pattern/matcher.hpp"

bool ov::pass::pattern::op::Or::match_value(Matcher* matcher,
                                            const Output<Node>& pattern_value,
                                            const Output<Node>& graph_value) {
    for (const auto& input_value : input_values()) {
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value, graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[input_value.get_node_shared_ptr()] = graph_value;
            return saved.finish(true);
        }
    }
    return false;
}
