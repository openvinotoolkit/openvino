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

// this is for more nuanced testing
class TestMatcher : public ngraph::pattern::Matcher
{
    using ngraph::pattern::Matcher::Matcher;

public:
    TestMatcher() {}
    bool virtual match_value(const ngraph::Output<ngraph::Node>& pattern_value,
                             const ngraph::Output<ngraph::Node>& graph_value) override
    {
        if (ngraph::is_type<::ngraph::op::Parameter>(pattern_value.get_node_shared_ptr()))
        {
            bool result = pattern_value == graph_value;
            if (result)
            {
                m_matched_list.push_back(graph_value.get_node_shared_ptr());
            }
            return result;
        }

        return this->ngraph::pattern::Matcher::match_value(pattern_value, graph_value);
    }

public:
    bool match(const std::shared_ptr<ngraph::Node>& pattern_node,
               const std::shared_ptr<ngraph::Node>& graph_node)
    {
        NGRAPH_CHECK(pattern_node && graph_node); // the same condition throws an exception in the
                                                  // non-test version of `match`
        NGRAPH_DEBUG << "Starting match pattern = " << pattern_node->get_name()
                     << " , graph_node = " << graph_node->get_name();

        m_pattern_node = pattern_node;
        return ngraph::pattern::Matcher::match(graph_node, ngraph::pattern::PatternValueMap{});
    }
};
