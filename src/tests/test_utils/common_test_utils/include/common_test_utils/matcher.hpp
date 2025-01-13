// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/util/log.hpp"

// this is for more nuanced testing
class TestMatcher : public ov::pass::pattern::Matcher {
    using ov::pass::pattern::Matcher::Matcher;

public:
    TestMatcher() = default;
    bool match_value(const ov::Output<ov::Node>& pattern_value, const ov::Output<ov::Node>& graph_value) override {
        if (ov::is_type<ov::op::v0::Parameter>(pattern_value.get_node_shared_ptr())) {
            bool result = pattern_value == graph_value;
            if (result) {
                m_matched_list.push_back(graph_value.get_node_shared_ptr());
            }
            return result;
        }

        return this->ov::pass::pattern::Matcher::match_value(pattern_value, graph_value);
    }

public:
    bool match(const std::shared_ptr<ov::Node>& pattern_node, const std::shared_ptr<ov::Node>& graph_node) {
        OPENVINO_ASSERT(pattern_node && graph_node);  // the same condition throws an exception in the
                                                      // non-test version of `match`
        OPENVINO_DEBUG("Starting match pattern = ",
                       pattern_node->get_name(),
                       " , graph_node = ",
                       graph_node->get_name());

        m_pattern_node = pattern_node;
        return ov::pass::pattern::Matcher::match(graph_node, ov::pass::pattern::PatternValueMap{});
    }
};
