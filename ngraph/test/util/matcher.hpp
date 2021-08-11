// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// this is for more nuanced testing
class TestMatcher : public ov::pattern::Matcher
{
    using ov::pattern::Matcher::Matcher;

public:
    TestMatcher() {}
    bool virtual match_value(const ov::Output<ov::Node>& pattern_value,
                             const ov::Output<ov::Node>& graph_value) override
    {
        if (ov::is_type<::ov::op::Parameter>(pattern_value.get_node_shared_ptr()))
        {
            bool result = pattern_value == graph_value;
            if (result)
            {
                m_matched_list.push_back(graph_value.get_node_shared_ptr());
            }
            return result;
        }

        return this->ov::pattern::Matcher::match_value(pattern_value, graph_value);
    }

public:
    bool match(const std::shared_ptr<ov::Node>& pattern_node,
               const std::shared_ptr<ov::Node>& graph_node)
    {
        NGRAPH_CHECK(pattern_node && graph_node); // the same condition throws an exception in the
                                                  // non-test version of `match`
        NGRAPH_DEBUG << "Starting match pattern = " << pattern_node->get_name()
                     << " , graph_node = " << graph_node->get_name();

        m_pattern_node = pattern_node;
        return ov::pattern::Matcher::match(graph_node, ov::pattern::PatternValueMap{});
    }
};
