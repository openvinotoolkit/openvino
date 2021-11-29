// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/convert_gather_v7_to_gather_v1.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertGather7toGather1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{2, 2});
        auto axis = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather_v7 = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_v7}, ngraph::ParameterVector{data, indices});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertGather7ToGather1>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{2, 2});
        auto axis = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather_v1 = std::make_shared<ngraph::opset1::Gather>(data, indices, axis);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_v1}, ngraph::ParameterVector{data, indices});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertGather7toGather1_nonzero_batch_dims) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{2, 2});
        auto axis = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});

        auto gather_v7 = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, -1);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_v7}, ngraph::ParameterVector{data, indices});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertGather7ToGather1>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    // if batch_dims != 0 Gather-7 must remain
    ASSERT_EQ(count_ops_of_type<ngraph::opset1::Gather>(f), 0);
    ASSERT_EQ(count_ops_of_type<ngraph::opset7::Gather>(f), 1);
}
