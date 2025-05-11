// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/any_output.hpp"

#include "openvino/pass/pattern/matcher.hpp"

bool ov::pass::pattern::op::AnyOutput::match_value(Matcher* matcher,
                                                   const Output<Node>& pattern_value,
                                                   const Output<Node>& graph_value) {
    return input_value(0).get_node()->match_node(matcher, graph_value);
}
