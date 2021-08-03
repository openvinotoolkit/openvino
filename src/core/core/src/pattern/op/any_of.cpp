// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/pattern/op/any_of.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::AnyOf::type_info;

const NodeTypeInfo& pattern::op::AnyOf::get_type_info() const
{
    return type_info;
}

bool pattern::op::AnyOf::match_value(Matcher* matcher,
                                     const Output<Node>& pattern_value,
                                     const Output<Node>& graph_value)
{
    matcher->add_node(graph_value);
    return m_predicate(graph_value) && ([&]() {
               for (auto arg : graph_value.get_node_shared_ptr()->input_values())
               {
                   auto saved = matcher->start_match();
                   if (matcher->match_value(input_value(0), arg))
                   {
                       return saved.finish(true);
                   }
               }
               return false;
           }());
}
