// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/optimize_gather_nd.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, OptimizeGatherND_2by2indices_validinputs) {
    {
        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 2}, {1, 0, 0, 0});
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});

        auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_2by2indices_invalidinputs) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 2}, {1, 0, 0, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_2by1indices_validinputs) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1}, {1, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {-1, 2});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_2by0indices_invalidinputs) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2});

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_2by0indices_validinputs) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 1});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {1});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_1leadingdim) {
    {
        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1, 1}, {1, 0});
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});

        auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {-1, 2});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1}, {1, 0});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_3leadingdims) {
    {
        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1, 1, 1, 1}, {1, 0});
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});

        auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {-1, 2});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1, 1, 1}, {1, 0});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_nonzerobatchdim) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1}, {1, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        size_t batch_dims = 1;

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices, batch_dims);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_zerobatchdim) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1}, {1, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        size_t batch_dims = 0;

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices, batch_dims);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {-1, 2});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_nonzerobatchdim_complexexample1) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1}, {1, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4});
        size_t batch_dims = 1;

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices, batch_dims);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_zerobatchdim_complexexample2) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 1}, {1, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4});
        size_t batch_dims = 0;

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices, batch_dims);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {-1, 3, 4});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3, 4});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {1, 0});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_complexexample3) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 2, 3}, {0, 0, 3, 0, 0, 1, 0, 0, 4, 0, 0, 2});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 5});

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {-1});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 5});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 2}, {3, 1, 4, 2});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, OptimizeGatherND_complexexample4) {
    {
        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 2, 2}, {0, 1, 0, 3, 0, 2, 0, 0});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 4, 1});

        const auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
    {
        const auto shape = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {-1, 1});
        const auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{1, 4, 1});
        const auto reshape = std::make_shared<ngraph::opset8::Reshape>(data, shape, true);

        const auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 2}, {1, 3, 2, 0});
        const auto axis = ngraph::opset8::Constant::create<int64_t>(ngraph::element::Type_t::i64, ngraph::Shape{}, {0});
        const auto gather = std::make_shared<ngraph::opset8::Gather>(reshape, indices, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather}, ngraph::ParameterVector{data});
    }
}
