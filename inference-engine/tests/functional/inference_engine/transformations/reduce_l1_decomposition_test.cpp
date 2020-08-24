// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/reduce_l1_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ReduceL1DecompositionTest) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto axes = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i32, ngraph::Shape{1});
        auto reduce_l1 = std::make_shared<ngraph::opset4::ReduceL1>(data, axes, true);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{reduce_l1}, ngraph::ParameterVector{data, axes});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ReduceL1Decomposition>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto axes = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i32, ngraph::Shape{1});
        auto abs = std::make_shared<ngraph::opset4::Abs>(data);
        auto reduce_l1 = std::make_shared<ngraph::opset4::ReduceSum>(abs, axes, true);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reduce_l1}, ngraph::ParameterVector{data, axes});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
