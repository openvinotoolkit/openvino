// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/move_eltwise_up_data_movement_per_channel.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

class MoveEltwiseUpThroughDataMovPerChannelGPUTest : public TransformationTestsF {
protected:
    void register_pass() {
        manager.register_pass<ov::intel_gpu::MoveEltwiseUpThroughDataMovPerChannel>();
    }
};

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, SqueezeMiddleAxisSquareDims) {
    const ov::Shape shape{1, 1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
        auto constant = v0::Constant::create(ov::element::f32, {1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(squeeze, constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        register_pass();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto constant = v0::Constant::create(ov::element::f32, {1, 1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(input, constant);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_axis);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, ReshapeReduceRankSquareDims) {
    const ov::Shape shape{1, 1, 5, 5};
    const std::vector<int64_t> target_shape{1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto reshape_pattern = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(input, reshape_pattern, false);
        auto constant = v0::Constant::create(ov::element::f32, {1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(reshape, constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        register_pass();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto constant = v0::Constant::create(ov::element::f32, {1, 1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(input, constant);
        auto reshape_pattern = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(add, reshape_pattern, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, SqueezeLowerRankConstant) {
    const ov::Shape shape{1, 1, 5, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
        auto constant = v0::Constant::create(ov::element::f32, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(squeeze, constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        register_pass();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto constant = v0::Constant::create(ov::element::f32, {1, 1, 1, 5}, {0.f, 1.f, 2.f, 3.f, 4.f});
        auto add = std::make_shared<v1::Add>(input, constant);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_axis);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, SqueezeTrailingAxisLegal) {
    const ov::Shape shape{10, 20, 1, 1};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
        auto constant = v0::Constant::create(ov::element::f32, {10, 1, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(squeeze, constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        register_pass();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto constant = v0::Constant::create(ov::element::f32, {10, 1, 1, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, constant);
        auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto squeeze = std::make_shared<v0::Squeeze>(add, squeeze_axis);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, UnsqueezeMultipleAxes) {
    const ov::Shape shape{10, 20};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto unsqueeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(input, unsqueeze_axes);
        auto constant = v0::Constant::create(ov::element::f32, {1, 20, 1, 1}, {0.5f});
        auto add = std::make_shared<v1::Add>(unsqueeze, constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        register_pass();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto constant = v0::Constant::create(ov::element::f32, {1, 20}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, constant);
        auto unsqueeze_axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(add, unsqueeze_axes);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, ReshapeLegalExpand) {
    const ov::Shape shape{1, 3, 20};
    const std::vector<int64_t> target_shape{1, 3, 4, 5};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto reshape_pattern = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(input, reshape_pattern, false);
        auto constant = v0::Constant::create(ov::element::f32, {1, 3, 1, 1}, {0.5f});
        auto multiply = std::make_shared<v1::Multiply>(reshape, constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{multiply}, ov::ParameterVector{input});
        register_pass();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto constant = v0::Constant::create(ov::element::f32, {1, 3, 1}, {0.5f});
        auto multiply = std::make_shared<v1::Multiply>(input, constant);
        auto reshape_pattern = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        auto reshape = std::make_shared<v1::Reshape>(multiply, reshape_pattern, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{reshape}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, SqueezeNoAxesInput) {
    const ov::Shape shape{10, 1, 20};
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto squeeze = std::make_shared<v0::Squeeze>(input);
        auto constant = v0::Constant::create(ov::element::f32, {1, 20}, {0.5f});
        auto add = std::make_shared<v1::Add>(squeeze, constant);

        model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
        register_pass();
    }
    {
        auto input = std::make_shared<v0::Parameter>(ov::element::f32, shape);
        auto constant = v0::Constant::create(ov::element::f32, {1, 1, 20}, {0.5f});
        auto add = std::make_shared<v1::Add>(input, constant);
        auto squeeze = std::make_shared<v0::Squeeze>(add);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{input});
    }
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, DynamicInputReject) {
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::PartialShape{1, -1, 5, 5});
    auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
    auto constant = v0::Constant::create(ov::element::f32, {1, 1, 5}, {0.5f});
    auto add = std::make_shared<v1::Add>(squeeze, constant);

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    register_pass();
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, ReshapePDPDAxisZeroReject) {
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{5, 2, 5});
    auto reshape_pattern = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{5, 2, 5});
    auto reshape = std::make_shared<v1::Reshape>(input, reshape_pattern, false);
    auto constant = v0::Constant::create(ov::element::f32, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
    auto add = std::make_shared<v1::Add>(reshape, constant, ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 0));

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    register_pass();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(MoveEltwiseUpThroughDataMovPerChannelGPUTest, SqueezePDPDAxisOneReject) {
    auto input = std::make_shared<v0::Parameter>(ov::element::f32, ov::Shape{2, 1, 5, 5});
    auto squeeze_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto squeeze = std::make_shared<v0::Squeeze>(input, squeeze_axis);
    auto constant = v0::Constant::create(ov::element::f32, {5}, {0.f, 1.f, 2.f, 3.f, 4.f});
    auto add = std::make_shared<v1::Add>(squeeze, constant, ov::op::AutoBroadcastSpec(ov::op::AutoBroadcastType::PDPD, 1));

    model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});
    register_pass();
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

}  // namespace
