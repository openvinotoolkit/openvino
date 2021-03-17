//*****************************************************************************
// Copyright 2017-2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#include "ngraph/pattern/op/wrap_type.hpp"
#include "ngraph/pattern/matcher.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::WrapType::type_info;

const NodeTypeInfo& pattern::op::WrapType::get_type_info() const
{
    return type_info;
}

bool pattern::op::WrapType::match_value(Matcher* matcher,
                                        const Output<Node>& pattern_value,
                                        const Output<Node>& graph_value)
{
    if (std::any_of(m_wrapped_types.begin(),
                    m_wrapped_types.end(),
                    [&](const NodeTypeInfo& type_info) {
                        return graph_value.get_node_shared_ptr()->get_type_info().is_castable(
                            type_info);
                    }) &&
        m_predicate(graph_value))
    {
        auto& pattern_map = matcher->get_pattern_value_map();
        pattern_map[shared_from_this()] = graph_value;
        matcher->add_node(graph_value);
        return (get_input_size() == 0
                    ? true
                    : matcher->match_arguments(pattern_value.get_node(),
                                               graph_value.get_node_shared_ptr()));
    }
    return false;
}

NodeTypeInfo pattern::op::WrapType::get_wrapped_type() const
{
    if (m_wrapped_types.size() > 1)
    {
        throw ngraph::ngraph_error("get_wrapped_type() called on WrapType with more than one type");
    }
    return m_wrapped_types.at(0);
}

const std::vector<NodeTypeInfo>& pattern::op::WrapType::get_wrapped_types() const
{
    return m_wrapped_types;
}