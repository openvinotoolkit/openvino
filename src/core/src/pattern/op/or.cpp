// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/or.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

#ifdef ENABLE_OPENVINO_DEBUG
using namespace ov::util;
#endif

bool ov::pass::pattern::op::Or::match_value(Matcher* matcher,
                                            const Output<Node>& pattern_value,
                                            const Output<Node>& graph_value) {
    OV_LOG_MATCHING(matcher, level_string(matcher->level), "├─ CHECKING ", this->get_input_size(), " OR BRANCHES: ", this->get_name());
    for (size_t i = 0; i < get_input_size(); ++i) {
        OV_LOG_MATCHING(matcher, level_string(++matcher->level));
        OV_LOG_MATCHING(matcher, level_string(matcher->level++), "{  BRANCH ", i, ": ", ov::node_version_type_str(input_value(i).get_node_shared_ptr()));
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value(i), graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[shared_from_this()] = graph_value;
            auto res = saved.finish(true);
            OV_LOG_MATCHING(matcher, level_string(--matcher->level), "│");
            OV_LOG_MATCHING(matcher, level_string(matcher->level--), "}  ", OV_GREEN, "BRANCH ", i, " MATCHED");
            OV_LOG_MATCHING(matcher, level_string(matcher->level), "│");
            OV_LOG_MATCHING(matcher, level_string(matcher->level), "}  ", OV_GREEN, "BRANCH ", i, " HAS MATCHED");
            return res;
        }
        OV_LOG_MATCHING(matcher, level_string(--matcher->level), "│");
        OV_LOG_MATCHING(matcher, level_string(matcher->level--), "}  ", OV_RED, "BRANCH ", i, " DIDN'T MATCH");
    }
    OV_LOG_MATCHING(matcher, level_string(matcher->level), "│");
    OV_LOG_MATCHING(matcher, level_string(matcher->level), "}  ", OV_RED, "NONE OF OR BRANCHES MATCHED");
    return false;
}
