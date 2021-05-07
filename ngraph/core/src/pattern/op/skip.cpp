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
