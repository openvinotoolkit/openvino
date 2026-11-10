// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin/transformations/convert_group_deconv_1d_to_2d.hpp"

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"

using namespace testing;
using namespace ov::intel_gpu;

namespace ov {
namespace test {
namespace intel_gpu {

namespace {

std::shared_ptr<ov::Model> make_1d(bool with_output_shape) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, -1});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 1, 1, 12}, {0.5f});
    std::shared_ptr<ov::Node> deconv;
    if (with_output_shape) {
        auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {64});
        deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            output_shape,
                                                                            ov::Strides{2},
                                                                            ov::CoordinateDiff{5},
                                                                            ov::CoordinateDiff{5},
                                                                            ov::Strides{1});
    } else {
        deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data,
                                                                            weights,
                                                                            ov::Strides{2},
                                                                            ov::CoordinateDiff{5},
                                                                            ov::CoordinateDiff{5},
                                                                            ov::Strides{1});
    }
    return std::make_shared<ov::Model>(ov::OutputVector{deconv}, ov::ParameterVector{data});
}

std::shared_ptr<ov::Model> make_1d_as_2d(bool with_output_shape) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, -1});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 1, 1, 12}, {0.5f});
    auto last_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    auto data_2d = std::make_shared<ov::op::v0::Unsqueeze>(data, last_axis);
    auto weights_2d = std::make_shared<ov::op::v0::Unsqueeze>(weights, last_axis);
    std::shared_ptr<ov::Node> deconv;
    if (with_output_shape) {
        auto output_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {64});
        auto unit = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        auto output_shape_2d = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{output_shape, unit}, 0);
        deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data_2d,
                                                                            weights_2d,
                                                                            output_shape_2d,
                                                                            ov::Strides{2, 1},
                                                                            ov::CoordinateDiff{5, 0},
                                                                            ov::CoordinateDiff{5, 0},
                                                                            ov::Strides{1, 1});
    } else {
        deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data_2d,
                                                                            weights_2d,
                                                                            ov::Strides{2, 1},
                                                                            ov::CoordinateDiff{5, 0},
                                                                            ov::CoordinateDiff{5, 0},
                                                                            ov::Strides{1, 1});
    }
    auto squeeze = std::make_shared<ov::op::v0::Squeeze>(deconv, last_axis);
    return std::make_shared<ov::Model>(ov::OutputVector{squeeze}, ov::ParameterVector{data});
}

}  // namespace

TEST_F(TransformationTestsF, ConvertGroupConvBackpropData1DTo2D) {
    model = make_1d(false);
    manager.register_pass<ConvertGroupConvBackpropData1DTo2D>();
    model_ref = make_1d_as_2d(false);
}

TEST_F(TransformationTestsF, ConvertGroupConvBackpropData1DTo2D_OutputShape) {
    model = make_1d(true);
    manager.register_pass<ConvertGroupConvBackpropData1DTo2D>();
    model_ref = make_1d_as_2d(true);
}

TEST_F(TransformationTestsF, ConvertGroupConvBackpropData1DTo2D_2D_NoOp) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 8, 16, 16});
    auto weights = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{8, 1, 1, 3, 3}, {0.5f});
    auto deconv = std::make_shared<ov::op::v1::GroupConvolutionBackpropData>(data,
                                                                             weights,
                                                                             ov::Strides{2, 2},
                                                                             ov::CoordinateDiff{0, 0},
                                                                             ov::CoordinateDiff{0, 0},
                                                                             ov::Strides{1, 1});
    model = std::make_shared<ov::Model>(ov::OutputVector{deconv}, ov::ParameterVector{data});
    manager.register_pass<ConvertGroupConvBackpropData1DTo2D>();
}

}  // namespace intel_gpu
}  // namespace test
}  // namespace ov
