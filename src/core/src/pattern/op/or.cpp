// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/or.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp" //TODO: maybe remove

bool ov::pass::pattern::op::Or::match_value(Matcher* matcher,
                                            const Output<Node>& pattern_value,
                                            const Output<Node>& graph_value) {
    OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "├─ CHECKING ", this->get_input_size(), " OR BRANCHES: ", get_name());
    for (size_t i = 0; i < get_input_size(); ++i) {
        matcher->level++;
        OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level));
        OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "┌─ BRANCH: ", i);
        matcher->level++;
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value(i), graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[shared_from_this()] = graph_value;
            auto res = saved.finish(true);
            matcher->level--;
            OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "│");
            OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "└─ BRANCH: ", i, ":", get_name()); // it would be ideal to print what branch matched (dunno if possible)
            matcher->level--;
            OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "│");
            OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "└─ ONE OF OR BRANCHES HAS BEEN MATCHED: ", i, ":", get_name()); // it would be ideal to print what branch matched (dunno if possible)
            return res;
        }
        matcher->level--;
        OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "│");
        OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "└─ DIDN'T MATCH BRANCH: ", i);
        matcher->level--;
    }
    OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "│");
    OPENVINO_DEBUG_EMPTY("[", matcher->get_name(), "] ", level_string(matcher->level), "└─ NONE OF OR BRANCHES MATCHED: ", get_name());
    return false;
}
