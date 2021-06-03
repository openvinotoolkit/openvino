// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/gather_normalize_negative_indices.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, GatherNegativeIndicesNormalize1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::GatherNegativeIndicesNormalize>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 15, 128});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});

        auto shape_of = std::make_shared<ngraph::opset7::ShapeOf>(data);
        auto input_gather = std::make_shared<ngraph::opset7::Gather>(shape_of,
            axis, ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0}));
        auto cast = std::make_shared<ngraph::opset7::Convert>(input_gather, ngraph::element::i32);
        auto add = std::make_shared<ngraph::opset7::Add>(cast, indices);
        auto gather = std::make_shared<ngraph::opset7::Gather>(data, add, axis);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, GatherNegativeIndicesNormalize_positive_ind) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::GatherNegativeIndicesNormalize>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, GatherNegativeIndicesNormalize_non_static_shape) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(2));
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis, 0);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::GatherNegativeIndicesNormalize>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(2));
        auto indices = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {-1});
        auto axis = ngraph::opset7::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {0});

        auto gather = std::make_shared<ngraph::opset7::Gather>(data, indices, axis);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
