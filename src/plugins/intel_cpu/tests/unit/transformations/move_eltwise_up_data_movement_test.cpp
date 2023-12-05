// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset8.hpp>
#include "ov_ops/type_relaxed.hpp"
#include <transformations/init_node_info.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/cpu_opset/common/pass/move_eltwise_up_data_movement.hpp>

using namespace testing;

class MoveEltwiseUpThroughDataMovTest: public TransformationTestsF{};

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwise) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(input);

        auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, TypeRelaxedEltwise) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        auto intermediate_op = std::make_shared<ov::opset8::Clamp>(input, 0, 6);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(intermediate_op, transpose_const);

        auto mul_const = ov::opset8::Constant::create(ov::element::f32, {}, {2.f});
        auto multiply = std::make_shared<ov::op::TypeRelaxed<ov::opset8::Multiply>>(transpose, mul_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{multiply}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        auto intermediate_op = std::make_shared<ov::opset8::Clamp>(input, 0, 6);

        auto mul_const = ov::opset8::Constant::create(ov::element::f32, {}, {2.f});
        auto multiply = std::make_shared<ov::op::TypeRelaxed<ov::opset8::Multiply>>(intermediate_op, mul_const);

        auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(multiply, transpose_const);

        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{transpose}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, EltwiseSequence) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;
    {
        auto input_left = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        auto input_right = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto matmul = std::make_shared<ov::opset8::MatMul>(input_left, input_right);

        auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(matmul, transpose_const);

        auto relu = std::make_shared<ov::opset8::Relu>(transpose);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(relu, unsqueeze_const);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{input_left, input_right});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input_left = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        auto input_right = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto matmul = std::make_shared<ov::opset8::MatMul>(input_left, input_right);

        auto relu = std::make_shared<ov::opset8::Relu>(matmul);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(relu);

        auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input_left, input_right});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, DataMovementTwoConsumers) {
    /* In this case transformation shouldn't apply */
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;

    auto input_left = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
    auto input_right = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

    auto matmul = std::make_shared<ov::opset8::MatMul>(input_left, input_right);

    auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
    auto transpose = std::make_shared<ov::opset8::Transpose>(matmul, transpose_const);

    auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

    auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);

    auto relu = std::make_shared<ov::opset8::Relu>(transpose);

    model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid, relu}, ov::ParameterVector{input_left, input_right});
    manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleBinaryEltwiseWithScalarOnSecondBranch) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto add = std::make_shared<ov::opset8::Add>(unsqueeze, ov::opset8::Constant::create(ov::element::f32, {}, {scalar_value}));

        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto add = std::make_shared<ov::opset8::Add>(input, ov::opset8::Constant::create(ov::element::f32, {}, {scalar_value}));

        auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(add, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleEltwiseWith5ScalarOnSecondBranch) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(input, unsqueeze_const);

        auto add = std::make_shared<ov::opset8::Add>(unsqueeze, ov::opset8::Constant::create(ov::element::f32, {1, 1, 1, 1, 1}, {scalar_value}));

        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto add = std::make_shared<ov::opset8::Add>(input, ov::opset8::Constant::create(ov::element::f32, {}, {scalar_value}));

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(add, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleBinaryEltwiseWithNotScalarOnSecondBranch) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;

    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

    auto transpose_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
    auto transpose = std::make_shared<ov::opset8::Transpose>(input, transpose_const);

    auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

    auto add_scalar = ov::opset8::Constant::create(ov::element::f32, {1, 1, 1, 3}, {0.5, 0.2, 0.3});
    auto add = std::make_shared<ov::opset8::Add>(unsqueeze, add_scalar);

    model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
    manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwiseDynamicShape) {
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(input, unsqueeze_const);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{input});
        manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
    }

    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(input);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(sigmoid, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwiseDynamicRank) {
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;

    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic(ov::Rank::dynamic()));

    auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(input, unsqueeze_const);
    auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);
    model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{input});
    manager.register_pass<ov::intel_cpu::MoveEltwiseUpThroughDataMov>();
}
