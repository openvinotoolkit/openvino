// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/pattern/op/true.hpp"

#include "openvino/pass/pattern/matcher.hpp"

bool ov::pass::pattern::op::True::match_value(Matcher* matcher,
                                              const Output<Node>& pattern_value,
                                              const Output<Node>& graph_value) {
    return true;
}
