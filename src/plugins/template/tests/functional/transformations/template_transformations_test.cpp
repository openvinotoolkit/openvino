// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

// ! [transformation:test]
TEST(TransformationTests, DISABLED_TemplateTest) {
    std::shared_ptr<ngraph::Function> f, f_ref;
    // f - ngraph::Function for applying transformation
    // f_ref - ngraph::Function that is expected after applying transformation
    {
        // Example function
        auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto divide = std::make_shared<ngraph::opset3::Divide>(data, divide_constant);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        // This transformation init runtime info attributes
        ngraph::pass::InitNodeInfo().run_on_model(f);

        // Run transformation
        // ngraph::pass::MyTransformation().run_on_function(f);

        // Check that after applying transformation all runtime info attributes was correctly propagated
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        // Example reference function
        auto data = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto pow = std::make_shared<ngraph::opset3::Power>(divide_constant,
                                                           ngraph::opset3::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {-1}));
        auto mul = std::make_shared<ngraph::opset3::Multiply>(data, pow);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});
    }

    // Compare that processed function and expected function are the same
    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
// ! [transformation:test]
