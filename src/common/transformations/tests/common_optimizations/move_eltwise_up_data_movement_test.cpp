// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/move_eltwise_up_data_movement.hpp"

#include <gtest/gtest.h>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/strided_slice.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/init_node_info.hpp"

using namespace testing;

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
class MoveEltwiseUpThroughDataMovTest : public TransformationTestsF {};

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwise) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(input, transpose_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

        auto sigmoid = std::make_shared<v0::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::OutputVector{sigmoid}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto sigmoid = std::make_shared<v0::Sigmoid>(input);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, TypeRelaxedEltwise) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto intermediate_op = std::make_shared<v0::Clamp>(input, 0, 6);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(intermediate_op, transpose_const);

        auto mul_const = v0::Constant::create(ov::element::f32, {}, {2.f});
        auto multiply = std::make_shared<ov::op::TypeRelaxed<v1::Multiply>>(transpose, mul_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{multiply}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto intermediate_op = std::make_shared<v0::Clamp>(input, 0, 6);

        auto mul_const = v0::Constant::create(ov::element::f32, {}, {2.f});
        auto multiply = std::make_shared<ov::op::TypeRelaxed<v1::Multiply>>(intermediate_op, mul_const);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(multiply, transpose_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{transpose}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, EltwiseSequence) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;
    {
        auto input_left = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto input_right = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto matmul = std::make_shared<v0::MatMul>(input_left, input_right);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(matmul, transpose_const);

        auto relu = std::make_shared<v0::Relu>(transpose);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(relu, unsqueeze_const);

        auto sigmoid = std::make_shared<v0::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::OutputVector{sigmoid}, ov::ParameterVector{input_left, input_right});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input_left = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto input_right = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto matmul = std::make_shared<v0::MatMul>(input_left, input_right);

        auto relu = std::make_shared<v0::Relu>(matmul);

        auto sigmoid = std::make_shared<v0::Sigmoid>(relu);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(sigmoid, transpose_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input_left, input_right});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, DataMovementTwoConsumers) {
    /* In this case transformation shouldn't apply */
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {1, 2, 0, 3};
    const int64_t unsqueeze_axis = 1;

    auto input_left = std::make_shared<v0::Parameter>(ov::element::f32, shape);
    auto input_right = std::make_shared<v0::Parameter>(ov::element::f32, shape);

    auto matmul = std::make_shared<v0::MatMul>(input_left, input_right);

    auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
    auto transpose = std::make_shared<v1::Transpose>(matmul, transpose_const);

    auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

    auto sigmoid = std::make_shared<v0::Sigmoid>(unsqueeze);

    auto relu = std::make_shared<v0::Relu>(transpose);

    model = std::make_shared<ov::Model>(ov::OutputVector{sigmoid, relu}, ov::ParameterVector{input_left, input_right});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleBinaryEltwiseWithScalarOnSecondBranch) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(input, transpose_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

        auto add = std::make_shared<v1::Add>(unsqueeze, v0::Constant::create(ov::element::f32, {}, {scalar_value}));

        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto add = std::make_shared<v1::Add>(input, v0::Constant::create(ov::element::f32, {}, {scalar_value}));

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(add, transpose_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleEltwiseWith5ScalarOnSecondBranch) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    const float scalar_value = 0.5f;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_const);

        auto add = std::make_shared<v1::Add>(unsqueeze,
                                             v0::Constant::create(ov::element::f32, {1, 1, 1, 1, 1}, {scalar_value}));

        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto add = std::make_shared<v1::Add>(input, v0::Constant::create(ov::element::f32, {}, {scalar_value}));

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(add, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleBinaryEltwiseWithNotScalarOnSecondBranch) {
    const ov::Shape shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;

    auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

    auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
    auto transpose = std::make_shared<v1::Transpose>(input, transpose_const);

    auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

    auto add_scalar = v0::Constant::create(ov::element::f32, {1, 1, 1, 3}, {0.5, 0.2, 0.3});
    auto add = std::make_shared<v1::Add>(unsqueeze, add_scalar);

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwiseDynamicShape) {
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_const);

        auto sigmoid = std::make_shared<v0::Sigmoid>(unsqueeze);

        model = std::make_shared<ov::Model>(ov::OutputVector{sigmoid}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }

    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(3));

        auto sigmoid = std::make_shared<v0::Sigmoid>(input);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(sigmoid, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SingleUnaryEltwiseDynamicRank) {
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;

    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape::dynamic(ov::Rank::dynamic()));

    auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
    auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_const);
    auto sigmoid = std::make_shared<v0::Sigmoid>(unsqueeze);
    model = std::make_shared<ov::Model>(ov::OutputVector{sigmoid}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, TransposeFakeQuantize) {
    const ov::Shape shape{1, 128, 12, 64};
    const std::vector<int64_t> input_order = {0, 2, 1, 3};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(input, transpose_const);
        auto fakequantize =
            std::make_shared<v0::FakeQuantize>(transpose,
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {-8.5}),
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {8.5}),
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {-128}),
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {127}),
                                               255);

        model = std::make_shared<ov::Model>(ov::OutputVector{fakequantize}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto fakequantize =
            std::make_shared<v0::FakeQuantize>(input,
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {-8.5}),
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {8.5}),
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {-128}),
                                               v0::Constant::create(ov::element::f32, ov::Shape{}, {127}),
                                               255);
        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(fakequantize, transpose_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{transpose}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, TransposeFakeQuantizePerChannel) {
    const ov::Shape shape{1, 12, 3, 64};
    const std::vector<int64_t> input_order = {0, 2, 1, 3};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(input, transpose_const);

        auto fakequantize = std::make_shared<v0::FakeQuantize>(
            transpose,
            v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {-8.5, -7.5, -10.}),
            v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, {8.5, 7.5, 10.}),
            v0::Constant::create(ov::element::f32, ov::Shape{}, {-128}),
            v0::Constant::create(ov::element::f32, ov::Shape{}, {127}),
            255);

        model = std::make_shared<ov::Model>(ov::OutputVector{fakequantize}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseUnsqueeze) {
    const ov::Shape shape{10, 20};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_const);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 20, 1, 1}, {0.5});
        auto add = std::make_shared<v1::Add>(unsqueeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 20}, {0.5});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(add, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseUnsqueezeReverseInOrder) {
    const ov::Shape shape{10, 20};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_const);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 20, 1, 1}, {0.5});
        auto add = std::make_shared<v1::Add>(per_channel_const, unsqueeze);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 20}, {0.5});
        auto add = std::make_shared<v1::Add>(per_channel_const, input);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});  // {10, 20, 1, 1}
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(add, unsqueeze_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

// The data movement op inserts an axis before the constant's channel axis, so the channel axis
// shifts. Matching on extent alone picks input axis 2 instead of axis 1 when both measure 5.
TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelUnsqueezeMiddleAxisSquareDims) {
    const ov::Shape shape{1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_axis);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 5, 1}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(unsqueeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 5, 1}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto unsqueeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(add, unsqueeze_axis);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseSqueeze) {
    const ov::Shape shape{10, 20, 1, 1};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto squeeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});  // {10, 20, 1}
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_const);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {10, 1, 1}, {0.5});
        auto add = std::make_shared<v1::Add>(squeeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {10, 1, 1, 1}, {0.5});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);

        auto squeeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});  // {10, 20, 1}
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_const);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseSqueezeIllegal_1) {
    // Only last dimensions can be updated by squeeze/unsqueeze op, while this subgraph removes dimension in the middle
    const ov::Shape shape{10, 1, 20};
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

    auto squeeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});  // {10, 20}
    auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_const);

    auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 20}, {0.5});
    auto add = std::make_shared<v1::Add>(squeeze, per_channel_const);

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelEltwiseSqueezeIllegal_2) {
    const ov::Shape shape{10, 20, 1, 1};
    // Data movement op with multiple consumers is not applicable
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

    auto squeeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});  // {10, 20, 1}
    auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_const);

    auto per_channel_const1 = v0::Constant::create(ov::element::f32, {10, 1, 1}, {0.5});
    auto add1 = std::make_shared<v1::Add>(squeeze, per_channel_const1);

    auto per_channel_const2 = v0::Constant::create(ov::element::f32, {10, 1, 1}, {0.5});
    auto add2 = std::make_shared<v1::Add>(squeeze, per_channel_const2);

    auto add3 = std::make_shared<v1::Add>(add1, add2);

    model = std::make_shared<ov::Model>(ov::OutputVector{add3}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelReshapeMultiply) {
    const ov::Shape shape{1, 3, 20};
    const std::vector<int64_t> target_shape = {1, 3, 4, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto reshape_constant =
            std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(input, reshape_constant, false);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 3, 1, 1}, {0.5});
        auto multiply = std::make_shared<v1::Multiply>(reshape, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{multiply}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);

        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 3, 1}, {0.5});
        auto multiply = std::make_shared<v1::Multiply>(input, per_channel_const);

        auto reshape_constant =
            std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(multiply, reshape_constant, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, SharedConstantNotReshapedForOtherConsumers) {
    const ov::Shape data_shape{1, 3, 224, 224};
    const std::vector<int64_t> input_order = {3, 2, 1, 0};
    const int64_t unsqueeze_axis = 2;
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::i64, data_shape);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(input, transpose_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

        auto shared_const = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

        auto multiply = std::make_shared<v1::Multiply>(unsqueeze, shared_const);

        auto slice_input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{4});
        auto begin = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto stride = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto slice = std::make_shared<v1::StridedSlice>(slice_input,
                                                        begin,
                                                        shared_const,
                                                        stride,
                                                        std::vector<int64_t>{0},
                                                        std::vector<int64_t>{0});

        model = std::make_shared<ov::Model>(ov::OutputVector{multiply, slice}, ov::ParameterVector{input, slice_input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::i64, data_shape);

        auto shared_const = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        auto scalar_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto multiply = std::make_shared<v1::Multiply>(input, scalar_const);

        auto transpose_const = v0::Constant::create(ov::element::i64, ov::Shape{input_order.size()}, input_order);
        auto transpose = std::make_shared<v1::Transpose>(multiply, transpose_const);

        auto unsqueeze_const = v0::Constant::create(ov::element::i64, ov::Shape{}, {unsqueeze_axis});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(transpose, unsqueeze_const);

        auto slice_input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{4});
        auto begin = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
        auto stride = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto slice = std::make_shared<v1::StridedSlice>(slice_input,
                                                        begin,
                                                        shared_const,
                                                        stride,
                                                        std::vector<int64_t>{0},
                                                        std::vector<int64_t>{0});

        model_ref =
            std::make_shared<ov::Model>(ov::OutputVector{unsqueeze, slice}, ov::ParameterVector{input, slice_input});
    }
}

// The data movement op drops an axis *before* the constant's channel axis, so the channel axis
// shifts. Matching on extent alone picks input axis 2 instead of axis 3 when both measure 5.
TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelSqueezeMiddleAxisSquareDims) {
    const ov::Shape shape{1, 1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(squeeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_axis);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelReshapeReduceRankSquareDims) {
    const ov::Shape shape{1, 1, 5, 5};
    const std::vector<int64_t> target_shape{1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto reshape_pattern =
            std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(input, reshape_pattern, false);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(reshape, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto reshape_pattern =
            std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(add, reshape_pattern, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// The constant has lower rank than the eltwise output and must be right-aligned before its
// channel axis can be mapped back through the Squeeze.
TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelSqueezeLowerRankConstant) {
    const ov::Shape shape{1, 1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(squeeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_axis);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelSqueezeNoAxesInput) {
    const ov::Shape shape{10, 1, 20};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze = std::make_shared<v0::Squeeze>(input);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 20}, {0.5f});
        auto add = std::make_shared<v1::Add>(squeeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 20}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto squeeze = std::make_shared<v0::Squeeze>(add);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelSqueezeDynamicDimension) {
    const ov::PartialShape shape{1, -1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 5}, {0.5f});
        auto add = std::make_shared<v1::Add>(squeeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 1, 5}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_axis);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelSqueezeDynamicTrailingDimensionsMultipleAxes) {
    const ov::PartialShape shape{1, -1, 1, 5, -1, 1};
    const std::vector<int64_t> axes{0, 2, -1};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axes);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 5, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(squeeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 1, 5, 1, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto squeeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_axes);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelUnsqueezeDynamicDimensionsNegativeAxes) {
    const ov::PartialShape shape{-1, 5, -1};
    const std::vector<int64_t> axes{0, -1};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto unsqueeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_axes);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 1, 5, 1, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(unsqueeze, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 5, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto unsqueeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(add, unsqueeze_axes);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelReshapeSymbolicTrailingDimension) {
    auto trailing_dim = ov::Dimension::dynamic();
    trailing_dim.set_symbol(std::make_shared<ov::Symbol>());
    const ov::PartialShape shape{1, 5, trailing_dim};
    const std::vector<int64_t> target_shape{5, 1, -1};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto reshape_pattern =
            std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(input, reshape_pattern, false);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {5, 1, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(reshape, per_channel_const);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto per_channel_const = v0::Constant::create(ov::element::f32, {1, 5, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, per_channel_const);
        auto reshape_pattern =
            std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(add, reshape_pattern, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelReshapeAnonymousDynamicTrailingDimensionReject) {
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, 5, -1});
    auto reshape_pattern = v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{5, 1, -1});
    auto reshape = std::make_shared<v1::Reshape>(input, reshape_pattern, false);
    auto per_channel_const = v0::Constant::create(ov::element::f32, {5, 1, 1}, {0.5f});
    auto add = std::make_shared<v1::Add>(reshape, per_channel_const);

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
}

// PDPD broadcasting anchors the constant at an explicit axis instead of right-aligning it, so the
// NumPy-based axis mapping does not apply.
TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelReshapePDPDAxisZeroReject) {
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{5, 2, 5});
    auto reshape_pattern =
        std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{5, 2, 5});
    auto reshape = std::make_shared<v1::Reshape>(input, reshape_pattern, false);
    auto per_channel_const = v0::Constant::create(ov::element::f32, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
    auto add = std::make_shared<v1::Add>(reshape,
                                         per_channel_const,
                                         ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0));

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovTest, PerChannelSqueezePDPDAxisOneReject) {
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{2, 1, 5, 5});
    auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
    auto per_channel_const = v0::Constant::create(ov::element::f32, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
    auto add = std::make_shared<v1::Add>(squeeze,
                                         per_channel_const,
                                         ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 1));

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    manager.register_pass<ov::pass::MoveEltwiseUpThroughDataMov>();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
