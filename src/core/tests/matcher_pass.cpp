// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>

#include "gtest/gtest.h"
#include "ngraph/graph_util.hpp"
#include "ngraph/log.hpp"
#include "ngraph/ngraph.hpp"
#include "ngraph/opsets/opset3.hpp"
#include "ngraph/pass/graph_rewrite.hpp"
#include "ngraph/pass/manager.hpp"

using namespace ngraph;
using namespace std;

OPENVINO_SUPPRESS_DEPRECATED_START

class TestMatcherPass : public pass::MatcherPass {
public:
    TestMatcherPass() {
        auto m_relu1 = ngraph::pattern::wrap_type<ngraph::opset3::Relu>(pattern::consumers_count(1));
        auto m_relu2 = ngraph::pattern::wrap_type<ngraph::opset3::Relu>({m_relu1});

        ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
            // Map that helps to connect labels with matched outputs
            auto& node_to_output = m.get_pattern_value_map();

            // Create new Relu operation and add register it for additional execution
            auto new_relu = register_new_node<ngraph::opset3::Relu>(
                node_to_output.at(m_relu1).get_node_shared_ptr()->input_value(0));

            // Copy runtime info attributes to newly created operation
            ngraph::copy_runtime_info(m.get_matched_nodes(), new_relu);

            // Save last Relu name to new Relu operation
            new_relu->set_friendly_name(m.get_match_root()->get_friendly_name());

            // Replace Relu->Relu with Relu
            ngraph::replace_node(m.get_match_root(), new_relu);

            // Return true as the root node was changed
            return true;
        };

        // Register pattern with Divide operation as a pattern root node
        auto m = std::make_shared<ngraph::pattern::Matcher>(m_relu2, "ReluReluFusion");
        // Register Matcher
        this->register_matcher(m, callback);
    }
};

TEST(pattern, matcher_pass) {
    {
        TestMatcherPass test_matcher;
        auto a = make_shared<opset3::Parameter>(element::f32, Shape{1});
        auto b = make_shared<opset3::Relu>(a);
        auto c = make_shared<opset3::Relu>(b);
        auto f = std::make_shared<Function>(ngraph::NodeVector{c}, ParameterVector{a});

        ASSERT_TRUE(test_matcher.get_matcher()->match(c->output(0)));
        ASSERT_TRUE(test_matcher.get_matcher()->get_matched_nodes().size() == 2);
        test_matcher.get_matcher()->clear_state();
        ASSERT_TRUE(test_matcher.get_matcher()->get_matched_nodes().empty());

        test_matcher.apply(c);
        ASSERT_TRUE(test_matcher.get_new_nodes().size() == 1);
        test_matcher.apply(test_matcher.get_new_nodes()[0]);
        ASSERT_TRUE(test_matcher.get_new_nodes().empty());
    }

    {
        TestMatcherPass test_matcher;
        auto a = make_shared<opset3::Parameter>(element::f32, Shape{1});
        auto b = make_shared<opset3::Relu>(a);
        auto c = make_shared<opset3::Relu>(b);
        auto f = std::make_shared<Function>(ngraph::NodeVector{b, c}, ParameterVector{a});

        ASSERT_FALSE(test_matcher.get_matcher()->match(c->output(0)));
    }

    {
        std::shared_ptr<Function> f;
        {
            auto a = make_shared<opset3::Parameter>(element::f32, Shape{1});
            auto b = make_shared<opset3::Relu>(a);
            auto c = make_shared<opset3::Relu>(b);
            auto d = make_shared<opset3::Relu>(c);
            f = std::make_shared<Function>(ngraph::NodeVector{d}, ParameterVector{a});
        }

        pass::GraphRewrite pass;
        pass.add_matcher<TestMatcherPass>();
        pass.run_on_function(f);

        // Parameter->Relu->Result
        ASSERT_TRUE(f->get_ops().size() == 3);
    }
}
