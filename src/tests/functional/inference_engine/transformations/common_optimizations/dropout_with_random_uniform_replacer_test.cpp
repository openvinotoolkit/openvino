// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <string>
#include <transformations/common_optimizations/dropout_with_random_uniform_replacer.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerCase1) {
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
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {30.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto broadcast_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.5});
        auto broadcast = std::make_shared<ngraph::opset8::Broadcast>(broadcast_const, input);

        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {30.0});
        auto add = std::make_shared<ngraph::opset8::Add>(broadcast, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerCase2) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f16,
                                                                  100,
                                                                  200);
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto broadcast_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {0.5});
        auto broadcast = std::make_shared<ngraph::opset8::Broadcast>(broadcast_const, input);

        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(broadcast, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerWithConvert) {
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
        auto convert = std::make_shared<ngraph::opset8::Convert>(ru, ngraph::element::f16);
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(convert, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto broadcast_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.5});
        auto broadcast = std::make_shared<ngraph::opset8::Broadcast>(broadcast_const, input);
        auto convert = std::make_shared<ngraph::opset8::Convert>(broadcast, ngraph::element::f16);

        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f16, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(convert, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}


TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerAddConstNegative) {
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
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.5});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
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
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.5});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}


TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerNonFloatRUNegative) {
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
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {10});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
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
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::i32, ngraph::Shape{}, {10});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerInvalidMinNegative) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-2.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-2.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerInvalidMaxNegative) {
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::i32, ngraph::Shape{3});
        auto min_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {0.0});
        auto max_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
        auto ru = std::make_shared<ngraph::opset8::RandomUniform>(input,
                                                                  min_const,
                                                                  max_const,
                                                                  ngraph::element::f32,
                                                                  100,
                                                                  200);
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}


TEST_F(TransformationTestsF, DropoutWithRandomUniformReplacerInvalidAddConstRankNegative) {
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
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {1.0, 2.0, 3.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::DropoutWithRandomUniformReplacer>();
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
        auto add_const = ngraph::opset8::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {1.0, 2.0, 3.0});
        auto add = std::make_shared<ngraph::opset8::Add>(ru, add_const);
        auto floor = std::make_shared<ngraph::opset8::Floor>(add);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{floor}, ngraph::ParameterVector{input});
    }
}