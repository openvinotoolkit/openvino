// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/or.hpp"

#include "openvino/core/log_util.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

bool ov::pass::pattern::op::Or::match_value(Matcher* matcher,
                                            const Output<Node>& pattern_value,
                                            const Output<Node>& graph_value) {
    OPENVINO_LOG_OR1(matcher, this->get_input_size(), this->get_name());
    for (size_t i = 0; i < get_input_size(); ++i) {
        OPENVINO_LOG_OR2(matcher, i, input_value(i));
        auto saved = matcher->start_match();
        if (matcher->match_value(input_value(i), graph_value)) {
            auto& pattern_map = matcher->get_pattern_value_map();
            pattern_map[shared_from_this()] = graph_value;
            auto res = saved.finish(true);
            OPENVINO_LOG_OR3(matcher, i);
            return res;
        }
        OPENVINO_LOG_OR4(matcher, i);
    }
    OPENVINO_LOG_OR5(matcher);
    return false;
}
