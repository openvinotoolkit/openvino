// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/placeholder.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/reduce_fc_dimensions.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

// Regular case, transformation should trigger
TEST_F(TransformationTestsF, ReduceFCDimensions1) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 16});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ReduceFCDimensions>();
    }
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 16});
        auto squeeze_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, {1, -1, 16});
        auto squeeze = std::make_shared<ov::op::v1::Reshape>(input1, squeeze_const, false);
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(squeeze, scale, no_bias);
        auto unsqueeze_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {1, 1, -1, 32});
        auto unsqueeze = std::make_shared<ov::op::v1::Reshape>(fc, unsqueeze_const, false);

        model_ref = std::make_shared<ov::Model>(ov::OutputVector{unsqueeze}, ov::ParameterVector{input1});
    }
}

// Incorrect input size, transformation should not trigger
TEST_F(TransformationTestsF, ReduceFCDimensions2) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 4, -1, 16});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ReduceFCDimensions>();
    }
    {
        model_ref = model->clone();
    }
}

// Bias present, transformation should not trigger
TEST_F(TransformationTestsF, ReduceFCDimensions3) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 16});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto bias = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1, 1, 32}, {1.0});
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ReduceFCDimensions>();
    }
    {
        model_ref = model->clone();
    }
}

// 3D weight, transformation should not trigger
TEST_F(TransformationTestsF, ReduceFCDimensions4) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 16});
        auto weights_const = ov::op::v0::Constant::create(ov::element::u8, ov::Shape{4, 32, 16}, {1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_const, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{4, 32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1});
        manager.register_pass<ReduceFCDimensions>();
    }
    {
        model_ref = model->clone();
    }
}

// Dynamic result dim, transformation should not trigger
TEST_F(TransformationTestsF, ReduceFCDimensions5) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, -1, 16});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{-1, 16});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f32);
        auto scale_param = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{-1, 1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_param);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param, scale_param});
        manager.register_pass<ReduceFCDimensions>();
    }
    {
        model_ref = model->clone();
    }
}

// Dynamic inner dim, transformation should not trigger
TEST_F(TransformationTestsF, ReduceFCDimensions6) {
    {
        auto input1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 1, 10, -1});
        auto weights_param = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{32, -1});
        auto convert = std::make_shared<ov::op::v0::Convert>(weights_param, ov::element::f32);
        auto scale_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1});
        auto scale = std::make_shared<ov::op::v1::Multiply>(convert, scale_const);
        auto no_bias = std::make_shared<ov::intel_gpu::op::Placeholder>();
        auto fc = std::make_shared<ov::intel_gpu::op::FullyConnected>(input1, scale, no_bias);

        model = std::make_shared<ov::Model>(ov::OutputVector{fc}, ov::ParameterVector{input1, weights_param});
        manager.register_pass<ReduceFCDimensions>();
    }
    {
        model_ref = model->clone();
    }
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
