// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/true.hpp"

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

#ifdef ENABLE_OPENVINO_DEBUG
using namespace ov::util;
#endif

bool ov::pass::pattern::op::True::match_value(Matcher* matcher,
                                              const Output<Node>& pattern_value,
                                              const Output<Node>& graph_value) {
    OV_LOG_MATCHING(matcher, matcher->level_str, "}  ", OV_GREEN, "TRUE ALWAYS MATCHES");
    return true;
}
