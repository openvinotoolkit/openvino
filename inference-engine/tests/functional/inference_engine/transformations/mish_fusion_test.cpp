// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/visualize_tree.hpp>
#include <transformations/mish_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, MishFusing) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input0 = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f64, ngraph::Shape{3, 1, 2});
        auto exp = std::make_shared<ngraph::opset4::Exp>(input0);
        auto input_const = ngraph::opset4::Constant::create(ngraph::element::f64, ngraph::Shape{1}, {-1});
        auto add = std::make_shared<ngraph::opset4::Add>(exp, input_const);
        auto log = std::make_shared<ngraph::opset4::Log>(add);
        auto tanh = std::make_shared<ngraph::opset4::Tanh>(log);
        auto mul = std::make_shared<ngraph::opset4::Multiply>(input0, tanh);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input0});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::MishFusion>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto mish = std::make_shared<ngraph::opset4::Mish>(data);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mish}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
