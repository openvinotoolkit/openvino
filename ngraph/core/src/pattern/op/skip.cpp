// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/skip.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::Skip::type_info;

const NodeTypeInfo& pattern::op::Skip::get_type_info() const
{
    return type_info;
}

bool pattern::op::Skip::match_value(Matcher* matcher,
                                    const Output<Node>& pattern_value,
                                    const Output<Node>& graph_value)
{
    matcher->add_node(graph_value);
    return m_predicate(graph_value) ? matcher->match_arguments(pattern_value.get_node(),
                                                               graph_value.get_node_shared_ptr())
                                    : matcher->match_value(input_value(0), graph_value);
}
