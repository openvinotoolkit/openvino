// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/or.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp" //TODO: maybe remove

bool ov::pass::pattern::op::Or::match_value(Matcher* matcher,
                                            const Output<Node>& pattern_value,
                                            const Output<Node>& graph_value) {
    OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "]         CHECKING ", this->get_input_size(), " OR BRANCHES");
    for (const auto& input_value : input_values()) {
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value, graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[shared_from_this()] = graph_value;
            return saved.finish(true);
        }
    }
    OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "]         -- NONE OF OR BRANCHES MATCHED");
    return false;
}
