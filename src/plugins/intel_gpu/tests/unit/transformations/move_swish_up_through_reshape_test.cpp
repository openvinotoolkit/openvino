// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/abs.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/move_swish_up_through_reshape.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace {
std::shared_ptr<ov::op::v1::Reshape> make_reshape(const ov::Output<ov::Node>& in) {
    auto shape = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, -1, 128});
    return std::make_shared<ov::op::v1::Reshape>(in, shape, false);
}
}  // namespace

// Swish(Reshape(x)) -> Reshape(Swish(x)) when the Reshape feeds only the Swish.
TEST_F(TransformationTestsF, MoveSwishUpThroughReshape) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 2048});
        auto swish = std::make_shared<ov::op::v4::Swish>(make_reshape(x));
        model = std::make_shared<ov::Model>(ov::OutputVector{swish}, ov::ParameterVector{x});
        manager.register_pass<MoveSwishUpThroughReshape>();
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 2048});
        auto swish = std::make_shared<ov::op::v4::Swish>(x);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{make_reshape(swish)}, ov::ParameterVector{x});
    }
}

// The optional beta input is carried over.
TEST_F(TransformationTestsF, MoveSwishUpThroughReshapeWithBeta) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 2048});
        auto beta = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.5f});
        auto swish = std::make_shared<ov::op::v4::Swish>(make_reshape(x), beta);
        model = std::make_shared<ov::Model>(ov::OutputVector{swish}, ov::ParameterVector{x});
        manager.register_pass<MoveSwishUpThroughReshape>();
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 2048});
        auto beta = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {1.5f});
        auto swish = std::make_shared<ov::op::v4::Swish>(x, beta);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{make_reshape(swish)}, ov::ParameterVector{x});
    }
}

// A Reshape with a second consumer would expose the un-activated values: nothing may change.
TEST_F(TransformationTestsF, MoveSwishUpThroughReshapeSharedReshapeUntouched) {
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 2048});
        auto reshape = make_reshape(x);
        auto swish = std::make_shared<ov::op::v4::Swish>(reshape);
        auto other = std::make_shared<ov::op::v0::Abs>(reshape);
        model = std::make_shared<ov::Model>(ov::OutputVector{swish, other}, ov::ParameterVector{x});
        manager.register_pass<MoveSwishUpThroughReshape>();
    }
    {
        auto x = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, 2048});
        auto reshape = make_reshape(x);
        auto swish = std::make_shared<ov::op::v4::Swish>(reshape);
        auto other = std::make_shared<ov::op::v0::Abs>(reshape);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{swish, other}, ov::ParameterVector{x});
    }
}
