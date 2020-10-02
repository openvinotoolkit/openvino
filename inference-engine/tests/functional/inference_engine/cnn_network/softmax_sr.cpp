// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <map>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/function.hpp>
#include <common_test_utils/ngraph_test_utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/smart_reshape/softmax_sr.hpp>


using namespace testing;

TEST(SmartReshapeSoftMaxTests, ReshapeSoftMaxReshapeTests) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});

        auto first_pattern = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto reshape = std::make_shared<ngraph::opset4::Reshape>(data, first_pattern, true);

        auto softmax = std::make_shared<ngraph::opset4::Softmax>(reshape, 1);

        auto pattern = ngraph::opset4::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {1, 2, 3});
        auto reshape_back = std::make_shared<ngraph::opset4::Reshape>(softmax, pattern, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_back}, ngraph::ParameterVector{data, first_pattern});

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ngraph::pass::ReshapeSoftMaxReshape>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }
    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});

        auto first_pattern = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i64, ngraph::Shape{2});
        auto reshape = std::make_shared<ngraph::opset4::Reshape>(data, first_pattern, true);

        auto softmax = std::make_shared<ngraph::opset4::Softmax>(reshape, 1);

        auto pattern = std::make_shared<ngraph::opset4::ShapeOf>(data);
        auto reshape_back = std::make_shared<ngraph::opset4::Reshape>(softmax, pattern, true);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_back}, ngraph::ParameterVector{data, first_pattern});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}