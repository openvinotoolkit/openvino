// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <transformations/utils/pass_param.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/graph_rewrite.hpp>
#include <ngraph/opsets/opset1.hpp>

using namespace ::testing;
using namespace std;
using namespace ngraph;

class TestTransformation : public ngraph::pass::GraphRewrite, public ngraph::pass::PassParam {
public:
    TestTransformation() : GraphRewrite() {
        auto divide = std::make_shared<ngraph::pattern::op::Label>(element::f32, Shape{}, pattern::has_class<opset1::Divide>());
        ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
            if (transformation_callback(m.get_match_root())) {
                auto relu = std::make_shared<ngraph::opset1::Relu>(m.get_match_root()->input_value(0));
                ngraph::replace_node(m.get_match_root(), relu);
                return true;
            }
            return false;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(divide, "TestMatcher");
        this->add_matcher(m, callback, ngraph::pass::PassProperty::CHANGE_DYNAMIC_STATE);
    }
};

class Anchor : public ngraph::pass::GraphRewrite {
public:
    Anchor() : GraphRewrite() {}
};

TEST(GraphRewriteTest, MultipleMatchers) {
    std::shared_ptr<Function> f;
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);
        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
    }

    auto anchor = std::make_shared<Anchor>();
    auto pass = std::make_shared<TestTransformation>();
    {
        pass->setCallback([](const std::shared_ptr<const Node> & node) -> bool {
            return (std::dynamic_pointer_cast<const opset1::Divide>(node) != nullptr);
        });
        anchor->copy_matchers(pass);
    }

    anchor->run_on_function(f);
    ASSERT_TRUE(ngraph::op::util::has_op_with_type<opset1::Relu>(f));
}
