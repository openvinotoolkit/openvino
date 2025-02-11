// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <list>
#include <memory>

#include "openvino/core/rt_info.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov;
using namespace ov::pass;
using namespace std;

class TestMatcherPass : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TestMatcherPass");
    TestMatcherPass() {
        auto m_relu1 = ov::pass::pattern::wrap_type<ov::op::v0::Relu>(pattern::consumers_count(1));
        auto m_relu2 = ov::pass::pattern::wrap_type<ov::op::v0::Relu>({m_relu1});

        ov::graph_rewrite_callback callback = [m_relu1, this](pattern::Matcher& m) {
            // Map that helps to connect labels with matched outputs
            auto& node_to_output = m.get_pattern_value_map();

            // Create new Relu operation and add register it for additional execution
            auto new_relu =
                register_new_node<ov::op::v0::Relu>(node_to_output.at(m_relu1).get_node_shared_ptr()->input_value(0));

            // Copy runtime info attributes to newly created operation
            ov::copy_runtime_info(m.get_matched_nodes(), new_relu);

            // Save last Relu name to new Relu operation
            new_relu->set_friendly_name(m.get_match_root()->get_friendly_name());

            // Replace Relu->Relu with Relu
            ov::replace_node(m.get_match_root(), new_relu);

            // Return true as the root node was changed
            return true;
        };

        // Register pattern with Divide operation as a pattern root node
        auto m = std::make_shared<ov::pass::pattern::Matcher>(m_relu2, "ReluReluFusion");
        // Register Matcher
        this->register_matcher(m, callback);
    }
};

TEST(pattern, matcher_pass) {
    {
        TestMatcherPass test_matcher;
        auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
        auto b = make_shared<op::v0::Relu>(a);
        auto c = make_shared<op::v0::Relu>(b);
        auto f = std::make_shared<Model>(ov::NodeVector{c}, ParameterVector{a});

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
        auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
        auto b = make_shared<op::v0::Relu>(a);
        auto c = make_shared<op::v0::Relu>(b);
        auto f = std::make_shared<Model>(ov::NodeVector{b, c}, ParameterVector{a});

        ASSERT_FALSE(test_matcher.get_matcher()->match(c->output(0)));
    }

    {
        std::shared_ptr<Model> f;
        {
            auto a = make_shared<op::v0::Parameter>(element::f32, Shape{1});
            auto b = make_shared<op::v0::Relu>(a);
            auto c = make_shared<op::v0::Relu>(b);
            auto d = make_shared<op::v0::Relu>(c);
            f = std::make_shared<Model>(ov::NodeVector{d}, ParameterVector{a});
        }

        pass::GraphRewrite pass;
        pass.add_matcher<TestMatcherPass>();
        pass.run_on_model(f);

        // Parameter->Relu->Result
        ASSERT_TRUE(f->get_ops().size() == 3);
    }
}
