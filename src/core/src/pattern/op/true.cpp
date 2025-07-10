// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/true.hpp"

#include "openvino/core/log_util.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

bool ov::pass::pattern::op::True::match_value(Matcher* matcher,
                                              const Output<Node>& pattern_value,
                                              const Output<Node>& graph_value) {
    OPENVINO_LOG_TRUE1(matcher);
    return true;
}
