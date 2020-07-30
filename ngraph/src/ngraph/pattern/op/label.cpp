//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
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

#include "ngraph/pattern/op/label.hpp"
#include "ngraph/pattern/matcher.hpp"
#include "ngraph/pattern/op/or.hpp"
#include "ngraph/pattern/op/true.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo pattern::op::Label::type_info;

const NodeTypeInfo& pattern::op::Label::get_type_info() const
{
    return type_info;
}

Output<Node> pattern::op::Label::wrap_values(const OutputVector& wrapped_values)
{
    switch (wrapped_values.size())
    {
    case 0: return make_shared<pattern::op::True>()->output(0);
    case 1: return wrapped_values[0];
    default: return make_shared<pattern::op::Or>(wrapped_values)->output(0);
    }
}

bool pattern::op::Label::match_value(Matcher* matcher,
                                     const Output<Node>& pattern_value,
                                     const Output<Node>& graph_value)
{
    if (m_predicate(graph_value))
    {
        auto& pattern_map = matcher->get_pattern_value_map();
        auto saved = matcher->start_match();
        matcher->add_node(graph_value);
        if (pattern_map.count(shared_from_this()))
        {
            return saved.finish(pattern_map[shared_from_this()] == graph_value);
        }
        else
        {
            pattern_map[shared_from_this()] = graph_value;
            return saved.finish(matcher->match_value(input_value(0), graph_value));
        }
    }
    return false;
}

std::shared_ptr<Node> pattern::any_input()
{
    return std::make_shared<pattern::op::Label>();
}