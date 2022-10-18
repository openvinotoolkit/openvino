// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include <ngraph_transformations/move_eltwise_up_data_movement.hpp>

using namespace testing;

class MoveEltwiseUpThroughDataMovTest: public TransformationTestsF{};

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwise) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(input);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, TypeRelaxedEltwise) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
        auto intermediate_op = std::make_shared<ngraph::opset8::Clamp>(input, 0, 6);

        auto transpose_const =
            ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(intermediate_op, transpose_const);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {2.f});
        auto multiply = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset8::Multiply>>(transpose, mul_const);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{multiply}, ngraph::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
        auto intermediate_op = std::make_shared<ngraph::opset8::Clamp>(input, 0, 6);

        auto mul_const = ngraph::opset8::Constant::create(ngraph::element::f32, {}, {2.f});
        auto multiply = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset8::Multiply>>(intermediate_op, mul_const);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(multiply, transpose_const);

        function_ref =
            std::make_shared<ngraph::Function>(ngraph::NodeVector{transpose}, ngraph::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, EltwiseSequence) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;
    {
        auto input_left = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
        auto input_right = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_left, input_right);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(matmul, transpose_const);

        auto relu = std::make_shared<ngraph::opset8::Relu>(transpose);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(relu, unsqueeze_const);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input_left, input_right});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input_left = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
        auto input_right = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_left, input_right);

        auto relu = std::make_shared<ngraph::opset8::Relu>(matmul);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(relu);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input_left, input_right});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, DataMovementTwoConsumers) {
    /* In this case transformation shouldn't apply */
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;

    auto input_left = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);
    auto input_right = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

    auto matmul = std::make_shared<ngraph::opset8::MatMul>(input_left, input_right);

    auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
    auto transpose = std::make_shared<ngraph::opset8::Transpose>(matmul, transpose_const);

    auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

    auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

    auto relu = std::make_shared<ngraph::opset8::Relu>(transpose);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid, relu}, ngraph::ParameterVector{input_left, input_right});
    manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleBinaryEltwiseWithScalarOnSecondBranch) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto add = std::make_shared<ngraph::opset8::Add>(unsqueeze, ngraph::opset8::Constant::create(ngraph::element::f32, {}, {scalar_value}));

        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto add = std::make_shared<ngraph::opset8::Add>(input, ngraph::opset8::Constant::create(ngraph::element::f32, {}, {scalar_value}));

        auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ngraph::opset8::Transpose>(add, transpose_const);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleEltwiseWith5ScalarOnSecondBranch) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(input, unsqueeze_const);

        auto add = std::make_shared<ngraph::opset8::Add>(unsqueeze, ngraph::opset8::Constant::create(ngraph::element::f32, {1, 1, 1, 1, 1}, {scalar_value}));

        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

        auto add = std::make_shared<ngraph::opset8::Add>(input, ngraph::opset8::Constant::create(ngraph::element::f32, {}, {scalar_value}));

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(add, unsqueeze_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleBinaryEltwiseWithNotScalarOnSecondBranch) {
    const ngraph::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;

    auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, shape);

    auto transpose_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{input_order.size()}, input_order);
    auto transpose = std::make_shared<ngraph::opset8::Transpose>(input, transpose_const);

    auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(transpose, unsqueeze_const);

    auto add_scalar = ngraph::opset8::Constant::create(ngraph::element::f32, {1, 1, 1, 3}, {0.5, 0.2, 0.3});
    auto add = std::make_shared<ngraph::opset8::Add>(unsqueeze, add_scalar);

    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{add}, ngraph::ParameterVector{input});
    manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwiseDynamicShape) {
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(input, unsqueeze_const);

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }

    {
        auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(3));

        auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(input);

        auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(sigmoid, unsqueeze_const);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{unsqueeze}, ngraph::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwiseDynamicRank) {
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;

    auto input = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(ngraph::Rank::dynamic()));

    auto unsqueeze_const = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<ngraph::opset8::Unsqueeze>(input, unsqueeze_const);
    auto sigmoid = std::make_shared<ngraph::opset8::Sigmoid>(unsqueeze);
    function = std::make_shared<ngraph::Function>(ngraph::NodeVector{sigmoid}, ngraph::ParameterVector{input});
    manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
}
