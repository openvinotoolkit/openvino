// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/log.hpp"
#include "ngraph/pattern/matcher.hpp"

// this is for more nuanced testing
class TestMatcher : public ngraph::pattern::Matcher {
    using ngraph::pattern::Matcher::Matcher;

public:
    TestMatcher() = default;
    bool match_value(const ngraph::Output<ngraph::Node>& pattern_value,
                     const ngraph::Output<ngraph::Node>& graph_value) override {
        if (ngraph::is_type<::ngraph::op::Parameter>(pattern_value.get_node_shared_ptr())) {
            bool result = pattern_value == graph_value;
            if (result) {
                m_matched_list.push_back(graph_value.get_node_shared_ptr());
            }
            return result;
        }

        return this->ngraph::pattern::Matcher::match_value(pattern_value, graph_value);
    }

public:
    bool match(const std::shared_ptr<ngraph::Node>& pattern_node, const std::shared_ptr<ngraph::Node>& graph_node) {
        NGRAPH_CHECK(pattern_node && graph_node);  // the same condition throws an exception in the
                                                   // non-test version of `match`
        NGRAPH_DEBUG << "Starting match pattern = " << pattern_node->get_name()
                     << " , graph_node = " << graph_node->get_name();

        m_pattern_node = pattern_node;
        return ngraph::pattern::Matcher::match(graph_node, ngraph::pattern::PatternValueMap{});
    }
};
