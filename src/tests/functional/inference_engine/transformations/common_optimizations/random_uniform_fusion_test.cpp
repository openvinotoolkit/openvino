// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/common_optimizations/random_uniform_fusion.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, RandomUniformMulFusing) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 1}, {30.0});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(ru, mul_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {30.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ru}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformAddFusing) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {30.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);

        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{1, 1, 1, 1}, {-10.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto ru_max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {20.0});
        auto ru_min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-10.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  ru_min_const,
                                                                  ru_max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ru}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformWithConvertMulFusing) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto conv = std::make_shared<ngraph::opset8::Convert>(ru, ngraph::element::f16);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {30.0});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(conv, mul_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {30.0});

        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto conv = std::make_shared<ngraph::opset8::Convert>(ru, ngraph::element::f16);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, RandomUniformWithConvertAddFusing) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {30.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto conv = std::make_shared<ngraph::opset8::Convert>(ru, ngraph::element::f16);

        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {-10.0});
        auto add = std::make_shared<ngraph::opset8::Add>(conv, add_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto ru_min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-10.0});
        auto ru_max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {20.0});

        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  ru_min_const,
                                                                  ru_max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto conv = std::make_shared<ngraph::opset8::Convert>(ru, ngraph::element::f16);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{input});
    }
}


TEST_F(TransformationTestsF, RandomUniformFusingInvalidRUType) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {100});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::i32,
                                                                  100,
                                                                  200);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {30});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(ru, mul_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {100});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::i32,
                                                                  100,
                                                                  200);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {30});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(ru, mul_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }
}


TEST_F(TransformationTestsF, RandomUniformFusingInvalidConstShape) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {30, 20, 15});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(ru, mul_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::RandomUniformFusion>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {30, 20, 15});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(ru, mul_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{input});
    }
}
