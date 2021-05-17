// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/convert_gather_v1_to_gather_v7.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertGather1toGather7) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{2, 2});
        auto axis = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather_v1 = std::make_shared<ngraph::opset1::Gather>(data, indices, axis);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_v1}, ngraph::ParameterVector{data, indices});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertGather1ToGather7>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{2, 2});
        auto axis = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather_v7 = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_v7}, ngraph::ParameterVector{data, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
