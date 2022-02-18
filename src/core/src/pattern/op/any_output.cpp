// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/any_output.hpp"

#include "ngraph/pattern/matcher.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::pass::pattern::op::AnyOutput);

bool ov::pass::pattern::op::AnyOutput::match_value(Matcher* matcher,
                                                   const Output<Node>& pattern_value,
                                                   const Output<Node>& graph_value) {
    return input_value(0).get_node()->match_node(matcher, graph_value);
}
