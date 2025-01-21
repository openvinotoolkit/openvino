// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/or.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp" //TODO: maybe remove

//todo remove or handle
#include "transformations/utils/utils.hpp"

bool ov::pass::pattern::op::Or::match_value(Matcher* matcher,
                                            const Output<Node>& pattern_value,
                                            const Output<Node>& graph_value) {
    OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level), "├─ CHECKING ", this->get_input_size(), " OR BRANCHES: ", this->get_name());
    for (size_t i = 0; i < get_input_size(); ++i) {
        OPENVINO_DEBUG_EMPTY(matcher, level_string(++matcher->level));
        OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level++), "┌─ BRANCH ", i, ": ", ov::node_version_type_str(input_value(i).get_node_shared_ptr()));
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value(i), graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[shared_from_this()] = graph_value;
            auto res = saved.finish(true);
            OPENVINO_DEBUG_EMPTY(matcher, level_string(--matcher->level), "│");
            OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level--), "└─ BRANCH ", i, " MATCHED");
            OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level), "│");
            OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level), "└─ BRANCH ", i, " HAS MATCHED");
            return res;
        }
        OPENVINO_DEBUG_EMPTY(matcher, level_string(--matcher->level), "│");
        OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level--), "└─ BRANCH ", i, " DIDN'T MATCH");
    }
    OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level), "│");
    OPENVINO_DEBUG_EMPTY(matcher, level_string(matcher->level), "└─ NONE OF OR BRANCHES MATCHED");
    return false;
}
