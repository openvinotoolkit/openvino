// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/any.hpp"

#include "ngraph/pattern/matcher.hpp"

using namespace std;

BWDCMP_RTTI_DEFINITION(ov::pass::pattern::op::Any);

bool ov::pass::pattern::op::Any::match_value(Matcher* matcher,
                                             const Output<Node>& pattern_value,
                                             const Output<Node>& graph_value) {
    matcher->add_node(graph_value);
    return m_predicate(graph_value) &&
           matcher->match_arguments(pattern_value.get_node(), graph_value.get_node_shared_ptr());
}
