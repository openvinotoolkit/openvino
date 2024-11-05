// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/move_eltwise_up_data_movement.hpp"

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/gen_pattern.hpp"

#include <openvino/itt.hpp>
#include "openvino/pass/visualize_tree.hpp"

using namespace testing;

class MoveEltwiseUpThroughDataMovTest : public TransformationTestsF {};

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwise) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(input);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
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
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        auto intermediate_op = std::make_shared<ov::opset8::Clamp>(input, 0, 6);

        auto mul_const = ov::opset8::Constant::create(ov::element::f32, {}, {2.f});
        auto multiply = std::make_shared<ov::op::TypeRelaxed<ov::opset8::Multiply>>(intermediate_op, mul_const);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(multiply, transpose_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{transpose}, ov::ParameterVector{input});
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

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(matmul, transpose_const);

        auto relu = std::make_shared<ov::opset8::Relu>(transpose);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(relu, unsqueeze_const);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{input_left, input_right});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input_left = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);
        auto input_right = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto matmul = std::make_shared<ov::opset8::MatMul>(input_left, input_right);

        auto relu = std::make_shared<ov::opset8::Relu>(matmul);

        auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(relu);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        model_ref =
            std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input_left, input_right});
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
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleBinaryEltwiseWithScalarOnSecondBranch) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(input, transpose_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(transpose, unsqueeze_const);

        auto add =
            std::make_shared<ov::opset8::Add>(unsqueeze,
                                              ov::opset8::Constant::create(ov::element::f32, {}, {scalar_value}));

        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto add =
            std::make_shared<ov::opset8::Add>(input,
                                              ov::opset8::Constant::create(ov::element::f32, {}, {scalar_value}));

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
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

        auto add = std::make_shared<ov::opset8::Add>(
            unsqueeze,
            ov::opset8::Constant::create(ov::element::f32, {1, 1, 1, 1, 1}, {scalar_value}));

        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto add =
            std::make_shared<ov::opset8::Add>(input,
                                              ov::opset8::Constant::create(ov::element::f32, {}, {scalar_value}));

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
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
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
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
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

    auto input =
        std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic(ov::Rank::dynamic()));

    auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(input, unsqueeze_const);
    auto sigmoid = std::make_shared<ov::opset8::Sigmoid>(unsqueeze);
    model = std::make_shared<ov::Model>(ov::NodeVector{sigmoid}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, TransposeFakeQuantize) {
    const ov::Shape shape{1, 128, 12, 64};
    const std::vector<int64_t> input_order = {0, 2, 1, 3};
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(input, transpose_const);
        auto fakequantize = std::make_shared<ov::opset8::FakeQuantize>(
            transpose,
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {-8.5}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {8.5}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {-128}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {127}),
            255);

        model = std::make_shared<ov::Model>(ov::NodeVector{fakequantize}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto fakequantize = std::make_shared<ov::opset8::FakeQuantize>(
            input,
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {-8.5}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {8.5}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {-128}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {127}),
            255);
        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(fakequantize, transpose_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{transpose}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, TransposeFakeQuantizePerChannel) {
    const ov::Shape shape{1, 12, 3, 64};
    const std::vector<int64_t> input_order = {0, 2, 1, 3};
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto transpose_const =
            ov::opset8::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<ov::opset8::Transpose>(input, transpose_const);

        auto fakequantize = std::make_shared<ov::opset8::FakeQuantize>(
            transpose,
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {-8.5, -7.5, -10.}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {8.5, 7.5, 10.}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {-128}),
            ov::opset8::Constant::create(ov::element::f32, ov::Shape{}, {127}),
            255);

        model = std::make_shared<ov::Model>(ov::NodeVector{fakequantize}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseUnsqueeze) {
    const ov::Shape shape{10, 20};
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(input, unsqueeze_const);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {1, 20, 1, 1}, {0.5});
        auto add = std::make_shared<ov::opset8::Add>(unsqueeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {1, 20}, {0.5});
        auto add = std::make_shared<ov::opset8::Add>(input, per_channel_const);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(add, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseUnsqueezeReverseInOrder) {
    const ov::Shape shape{10, 20};
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(input, unsqueeze_const);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {1, 20, 1, 1}, {0.5});
        auto add = std::make_shared<ov::opset8::Add>(per_channel_const, unsqueeze);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {1, 20}, {0.5});
        auto add = std::make_shared<ov::opset8::Add>(per_channel_const, input);

        auto unsqueeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<ov::opset8::Unsqueeze>(add, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseSqueeze) {
    const ov::Shape shape{10, 20, 1, 1};
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto squeeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {2});  // {10, 20, 1}
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(input, squeeze_const);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {10, 1, 1}, {0.5});
        auto add = std::make_shared<ov::opset8::Add>(squeeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {10, 1, 1, 1}, {0.5});
        auto add = std::make_shared<ov::opset8::Add>(input, per_channel_const);

        auto squeeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {2});  // {10, 20, 1}
        auto squeeze = std::make_shared<ov::op::v0::Squeeze>(add, squeeze_const);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{squeeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseSqueezeIllegal_1) {
    // Only last dimensions can be updated by squeeze/unsqueeze op, while this subgraph removes dimension in the middle
    const ov::Shape shape{10, 1, 20};
    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

    auto squeeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {1});  // {10, 20}
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(input, squeeze_const);

    auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {1, 1, 20}, {0.5});
    auto add = std::make_shared<ov::opset8::Add>(squeeze, per_channel_const);

    model = std::make_shared<ov::Model>(ov::NodeVector{add}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseSqueezeIllegal_2) {
    const ov::Shape shape{10, 20, 1, 1};
    // Data movement op with multiple consumers is not applicable
    auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

    auto squeeze_const = ov::opset8::Constant::create(ov::element::i64, ov::Shape{}, {2});  // {10, 20, 1}
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(input, squeeze_const);

    auto per_channel_const1 = ov::opset8::Constant::create(ov::element::f32, {10, 1, 1}, {0.5});
    auto add1 = std::make_shared<ov::opset8::Add>(squeeze, per_channel_const1);

    auto per_channel_const2 = ov::opset8::Constant::create(ov::element::f32, {10, 1, 1}, {0.5});
    auto add2 = std::make_shared<ov::opset8::Add>(squeeze, per_channel_const2);

    auto add3 = std::make_shared<ov::opset8::Add>(add1, add2);

    model = std::make_shared<ov::Model>(ov::NodeVector{add3}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelReshapeMultiply) {
    const ov::Shape shape{1, 3, 20};
    const std::vector<int64_t> target_shape = {1, 3, 4, 5};
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto reshape_constant =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(input, reshape_constant, false);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {1, 3, 1, 1}, {0.5});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(reshape, per_channel_const);

        model = std::make_shared<ov::Model>(ov::NodeVector{multiply}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset8::Parameter>(ov::element::f32, shape);

        auto per_channel_const = ov::opset8::Constant::create(ov::element::f32, {1, 3, 1}, {0.5});
        auto multiply = std::make_shared<ov::op::v1::Multiply>(input, per_channel_const);

        auto reshape_constant =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<ov::op::v1::Reshape>(multiply, reshape_constant, false);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{reshape}, ov::ParameterVector{input});
    }
}


class TRANSFORMATIONS_API InnerSamplePass : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("InnerSamplePass", "0");
    InnerSamplePass() {
        // MATCHER_SCOPE(InnerSamplePass);
        auto matcher_name = "InnerSamplePass";
        auto eltwise_pattern = ov::pass::pattern::wrap_type<ov::op::util::UnaryElementwiseArithmetic,
                                                        ov::op::util::BinaryElementwiseArithmetic,
                                                        ov::op::v0::FakeQuantize>(ov::pass::pattern::has_static_rank());

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            std::cout << "found something" << std::endl;
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(eltwise_pattern, matcher_name);
        register_matcher(m, callback);
    }
};

class TRANSFORMATIONS_API SamplePass : public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("SamplePass", "0");
    SamplePass() {
        std::cout << "REGISTERING SamplePass" << std::endl;
        this->add_matcher<InnerSamplePass>();
    }
};

static inline void update_tensor_type(ov::Output<ov::Node> output, ov::element::Type_t new_type) {
    output.set_tensor_ptr(std::make_shared<ov::descriptor::Tensor>(new_type, output.get_tensor().get_partial_shape(), output.get_tensor().get_names()));
}

TEST_F(MoveEltwiseUpThroughDataMovTest, MoveThruTwoNodes) {
    const ov::Shape shape{15876, 1, 1, 1};
    const std::vector<int64_t> target_shape = {1, 3, 4, 5};
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876, 1, 1, 1});
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876, 1, 1, 1});

        auto add = ov::gen_pattern::makeOP<ov::opset1::Add>({input, input1}, {{"auto_broadcast", "numpy"}});

        auto reshape_const = ov::gen_pattern::makeConst(ov::element::i32, {1}, {15876});
        // auto reshape = ov::gen_pattern::makeOP<ov::opset1::Reshape>({input, reshape_const}, {{"special_zero", false}});
        auto reshape = ov::gen_pattern::makeOP<ov::opset1::Reshape>({add, reshape_const}, {{"special_zero", false}});

        auto fq_cons_0 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {0});
        auto fq_cons_1 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {2.53667});
        auto fq_cons_2 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {0});
        auto fq_cons_3 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {65504});

        auto fq = ov::gen_pattern::makeOP<ov::opset1::FakeQuantize>({reshape, fq_cons_0, fq_cons_1, fq_cons_2, fq_cons_3}, {{"levels", 256}, {"auto_broadcast", "numpy"}});

        auto sqrt = ov::gen_pattern::makeOP<ov::opset1::Sqrt>({fq}, {}); 

        auto another_fq_cons_0 = ov::gen_pattern::makeConst(ov::element::f16, {1}, {0});
        auto another_fq_cons_1 = ov::gen_pattern::makeConst(ov::element::f16, {1}, {278.75});
        auto another_fq_cons_2 = ov::gen_pattern::makeConst(ov::element::f16, {}, {0});
        auto another_fq_cons_3 = ov::gen_pattern::makeConst(ov::element::f16, {}, {255});

        auto another_fq = ov::gen_pattern::makeOP<ov::opset1::FakeQuantize>({sqrt, another_fq_cons_0, another_fq_cons_1, another_fq_cons_2, another_fq_cons_3}, {{"levels", 256}, {"auto_broadcast", "numpy"}});

        auto another_reshape_const = ov::gen_pattern::makeConst(ov::element::i32, {4}, {-1, 1, 1, 1});
        auto another_reshape = ov::gen_pattern::makeOP<ov::opset1::Reshape>({another_fq, another_reshape_const}, {{"special_zero", false}});

        auto mul_const = ov::gen_pattern::makeConst(ov::element::f32, {}, {1.09314});
        auto mul = ov::gen_pattern::makeOP<ov::opset1::Multiply>({another_reshape, mul_const}, {{"auto_broadcast", "numpy"}});

        auto pow_const = ov::gen_pattern::makeConst(ov::element::f32, {}, {-1});
        auto pow = ov::gen_pattern::makeOP<ov::opset1::Power>({mul, pow_const}, {{"auto_broadcast", "numpy"}});

        auto another_mul_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876,256,3,3});
        auto another_mul = ov::gen_pattern::makeOP<ov::opset1::Multiply>({another_mul_input, pow}, {{"auto_broadcast", "numpy"}});

        auto res = ov::gen_pattern::makeOP<ov::opset1::Result>({another_mul}, {});

        model = std::make_shared<ov::Model>(ov::NodeVector{res}, ov::ParameterVector{input, another_mul_input, input1});

        // update_tensor_type(input->output(0), ov::element::u8);
        update_tensor_type(add->output(0), ov::element::u8);

        update_tensor_type(reshape_const->output(0), ov::element::i32);
        update_tensor_type(reshape->output(0), ov::element::u8);

        update_tensor_type(fq->output(0), ov::element::f16);

        update_tensor_type(sqrt->output(0), ov::element::f16);
        
        update_tensor_type(another_fq->output(0), ov::element::u8);

        update_tensor_type(another_reshape_const->output(0), ov::element::i32);
        update_tensor_type(another_reshape->output(0), ov::element::u8);

        update_tensor_type(mul_const->output(0), ov::element::f32);
        update_tensor_type(mul->output(0), ov::element::f16);

        update_tensor_type(pow_const->output(0), ov::element::f16);
        update_tensor_type(pow->output(0), ov::element::f16);

        update_tensor_type(another_mul_input->output(0), ov::element::f16);
        update_tensor_type(another_mul->output(0), ov::element::f16);

        update_tensor_type(res->output(0), ov::element::f32);

        manager.set_per_pass_validation(false);
        manager.register_pass<ov::pass::VisualizeTree>("test_before.svg");
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
        manager.register_pass<ov::pass::VisualizeTree>("test_after_one.svg");

        std::cout << mul->input(0).get_partial_shape().rank() << std::endl;
    }
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::Shape{15876, 1, 1, 1});

        // auto mul_const = ov::opset1::Constant::create(ov::element::f32, {}, {1.09314});
        // auto mul = std::make_shared<ov::opset1::Multiply>(input, mul_const);

        // auto reshape_constant =
        //     ov::opset1::Constant::create(ov::element::i32, {1}, {15876});
        // auto reshape = std::make_shared<ov::opset1::Reshape>(mul, reshape_constant, false);

        // auto another_reshape_constant =
        //     ov::opset1::Constant::create(ov::element::i32, {4}, {-1, 1, 1, 1});
        // auto another_reshape = std::make_shared<ov::opset1::Reshape>(reshape, another_reshape_constant, false);

        // auto res = std::make_shared<ov::opset1::Result>(another_reshape);
        auto res = std::make_shared<ov::opset1::Result>(input);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{res}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, MoveThruTwoNodes2) {
    const ov::Shape shape{15876, 1, 1, 1};
    const std::vector<int64_t> target_shape = {1, 3, 4, 5};
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876, 1, 1, 1});
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876, 1, 1, 1});

        auto add = ov::gen_pattern::makeOP<ov::opset1::Add>({input, input1}, {{"auto_broadcast", "numpy"}});

        auto reshape_const = ov::gen_pattern::makeConst(ov::element::i32, {1}, {15876});
        auto reshape = ov::gen_pattern::makeOP<ov::opset1::Reshape>({add, reshape_const}, {{"special_zero", false}});

        auto fq_cons_0 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {0});
        auto fq_cons_1 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {2.53667});
        auto fq_cons_2 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {0});
        auto fq_cons_3 = ov::gen_pattern::makeConst(ov::element::f32, {1}, {65504});

        auto fq = ov::gen_pattern::makeOP<ov::opset1::FakeQuantize>({reshape, fq_cons_0, fq_cons_1, fq_cons_2, fq_cons_3}, {{"levels", 256}, {"auto_broadcast", "numpy"}});

        auto sqrt = ov::gen_pattern::makeOP<ov::opset1::Sqrt>({fq}, {}); 

        auto another_fq_cons_0 = ov::gen_pattern::makeConst(ov::element::f16, {1}, {0});
        auto another_fq_cons_1 = ov::gen_pattern::makeConst(ov::element::f16, {1}, {278.75});
        auto another_fq_cons_2 = ov::gen_pattern::makeConst(ov::element::f16, {}, {0});
        auto another_fq_cons_3 = ov::gen_pattern::makeConst(ov::element::f16, {}, {255});

        auto another_fq = ov::gen_pattern::makeOP<ov::opset1::FakeQuantize>({sqrt, another_fq_cons_0, another_fq_cons_1, another_fq_cons_2, another_fq_cons_3}, {{"levels", 256}, {"auto_broadcast", "numpy"}});

        auto another_reshape_const = ov::gen_pattern::makeConst(ov::element::i32, {4}, {-1, 1, 1, 1});
        auto another_reshape = ov::gen_pattern::makeOP<ov::opset1::Reshape>({another_fq, another_reshape_const}, {{"special_zero", false}});

        auto mul_const = ov::gen_pattern::makeConst(ov::element::f32, {}, {1.09314});
        auto mul = ov::gen_pattern::makeOP<ov::opset1::Multiply>({another_reshape, mul_const}, {{"auto_broadcast", "numpy"}});

        auto pow_const = ov::gen_pattern::makeConst(ov::element::f32, {}, {-1});
        auto pow = ov::gen_pattern::makeOP<ov::opset1::Power>({mul, pow_const}, {{"auto_broadcast", "numpy"}});

        auto another_mul_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876,256,3,3});
        auto another_mul = ov::gen_pattern::makeOP<ov::opset1::Multiply>({another_mul_input, pow}, {{"auto_broadcast", "numpy"}});

        auto res = ov::gen_pattern::makeOP<ov::opset1::Result>({another_mul}, {});

        model = std::make_shared<ov::Model>(ov::NodeVector{res}, ov::ParameterVector{input, another_mul_input, input1});

        manager.set_per_pass_validation(false);
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876, 1, 1, 1});
        auto input1 = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876, 1, 1, 1});

        auto add = ov::gen_pattern::makeOP<ov::opset1::Add>({input, input1}, {{"auto_broadcast", "numpy"}});

        auto fq_cons_0 = ov::gen_pattern::makeConst(ov::element::f32, {}, {0});
        auto fq_cons_1 = ov::gen_pattern::makeConst(ov::element::f32, {}, {2.53667});
        auto fq_cons_2 = ov::gen_pattern::makeConst(ov::element::f32, {}, {0});
        auto fq_cons_3 = ov::gen_pattern::makeConst(ov::element::f32, {}, {65504});

        auto fq = ov::gen_pattern::makeOP<ov::opset1::FakeQuantize>({add, fq_cons_0, fq_cons_1, fq_cons_2, fq_cons_3}, {{"levels", 256}, {"auto_broadcast", "numpy"}});

        auto sqrt = ov::gen_pattern::makeOP<ov::opset1::Sqrt>({fq}, {}); 

        auto another_fq_cons_0 = ov::gen_pattern::makeConst(ov::element::f16, {}, {0});
        auto another_fq_cons_1 = ov::gen_pattern::makeConst(ov::element::f16, {}, {278.75});
        auto another_fq_cons_2 = ov::gen_pattern::makeConst(ov::element::f16, {}, {0});
        auto another_fq_cons_3 = ov::gen_pattern::makeConst(ov::element::f16, {}, {255});

        auto another_fq = ov::gen_pattern::makeOP<ov::opset1::FakeQuantize>({sqrt, another_fq_cons_0, another_fq_cons_1, another_fq_cons_2, another_fq_cons_3}, {{"levels", 256}, {"auto_broadcast", "numpy"}});

        auto mul_const = ov::gen_pattern::makeConst(ov::element::f32, {}, {1.09314});
        auto mul = ov::gen_pattern::makeOP<ov::opset1::Multiply>({another_fq, mul_const}, {{"auto_broadcast", "numpy"}});

        auto pow_const = ov::gen_pattern::makeConst(ov::element::f32, {}, {-1});
        auto pow = ov::gen_pattern::makeOP<ov::opset1::Power>({mul, pow_const}, {{"auto_broadcast", "numpy"}});

        auto reshape_const = ov::gen_pattern::makeConst(ov::element::i32, {1}, {15876});
        auto reshape = ov::gen_pattern::makeOP<ov::opset1::Reshape>({pow, reshape_const}, {{"special_zero", false}});

        auto another_reshape_const = ov::gen_pattern::makeConst(ov::element::i32, {4}, {-1, 1, 1, 1});
        auto another_reshape = ov::gen_pattern::makeOP<ov::opset1::Reshape>({reshape, another_reshape_const}, {{"special_zero", false}});

        auto another_mul_input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15876,256,3,3});
        auto another_mul = ov::gen_pattern::makeOP<ov::opset1::Multiply>({another_mul_input, another_reshape}, {{"auto_broadcast", "numpy"}});

        auto res = ov::gen_pattern::makeOP<ov::opset1::Result>({another_mul}, {});

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{res}, ov::ParameterVector{input, input1, another_mul_input});
    }
}
