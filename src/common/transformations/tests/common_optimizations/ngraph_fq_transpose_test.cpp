// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph/op/transpose.hpp>
#include <ngraph/op/fake_quantize.hpp>
#include <transformations/common_optimizations/pull_transpose_through_fq.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, FQTransposeTest1) {
    {
        auto data = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 3}, {1, 2, 3});
        auto input_low = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2});
        auto input_high = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3});
        auto output_low = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2});
        auto output_high = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3});
        auto transpose_order = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});

        auto fq = std::make_shared<ngraph::op::FakeQuantize>(data, input_low, input_high, output_low, output_high, 1);
        auto transpose = std::make_shared<ngraph::op::Transpose>(fq, transpose_order);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose}, ngraph::ParameterVector{});

        manager.register_pass<ngraph::pass::PullTransposeThroughFQUp>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }
    {
        auto data = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3, 1}, {1, 2, 3});
        auto input_low = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 1}, {2});
        auto input_high = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 1}, {3});
        auto output_low = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 1}, {2});
        auto output_high = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 1}, {3});

        auto fq = std::make_shared<ngraph::op::FakeQuantize>(data, input_low, input_high, output_low, output_high, 1);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{fq}, ngraph::ParameterVector{});
    }
}

TEST_F(TransformationTestsF, FQTransposeNegativeCase) {
    auto create_graph = []() -> std::shared_ptr<ngraph::Function> {
        auto data = std::make_shared<ngraph::op::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 3, 1});
        auto sigmoid = std::make_shared<ngraph::op::Sigmoid>(data);
        auto input_low = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2});
        auto input_high = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3});
        auto output_low = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {2});
        auto output_high = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {3});
        auto transpose_order = ngraph::op::Constant::create(ngraph::element::i64, ngraph::Shape{3}, {0, 2, 1});

        auto fq = std::make_shared<ngraph::op::FakeQuantize>(sigmoid, input_low, input_high, output_low, output_high, 1);
        auto transpose = std::make_shared<ngraph::op::Transpose>(fq, transpose_order);

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose}, ngraph::ParameterVector{data});
    };
    function = create_graph();

    manager.register_pass<ngraph::pass::InitNodeInfo>();
    manager.register_pass<ngraph::pass::PullTransposeThroughFQUp>();
    manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
        check_rt_info(f);
    });

    function_ref = create_graph();
}
