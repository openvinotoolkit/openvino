// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convolution_to_matmul.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;

namespace {
std::shared_ptr<Model> create_conv_function(const Shape& input_shape,
                                            const Shape& weights_shape,
                                            bool with_dequantization) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);

    std::shared_ptr<Node> weights;
    if (with_dequantization) {
        auto weights_const = op::v0::Constant::create(element::i4, weights_shape, {1});
        auto weights_convert = std::make_shared<op::v0::Convert>(weights_const, element::f32);
        auto sub_const = op::v0::Constant::create(element::i4, {1}, {1});
        auto sub_convert = std::make_shared<op::v0::Convert>(sub_const, element::f32);
        auto subtract = std::make_shared<op::v1::Subtract>(weights_convert, sub_convert);
        auto mul_const = op::v0::Constant::create(element::f32, {1}, {2.0});
        weights = std::make_shared<op::v1::Multiply>(subtract, mul_const);
    } else {
        weights = op::v0::Constant::create(element::f32, weights_shape, {1.0});
    }

    auto conv = std::make_shared<op::v1::Convolution>(input,
                                                      weights,
                                                      Strides{1, 1},
                                                      CoordinateDiff{0, 0},
                                                      CoordinateDiff{0, 0},
                                                      Strides{1, 1});

    return std::make_shared<Model>(OutputVector{conv}, ParameterVector{input});
}

std::shared_ptr<Model> create_matmul_function(const Shape& input_shape,
                                              const Shape& weights_shape,
                                              const std::vector<int64_t>& input_transpose_order,
                                              const std::vector<int64_t>& output_transpose_order) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);

    auto weights_const = op::v0::Constant::create(element::i4, weights_shape, {1});
    auto weights_convert = std::make_shared<op::v0::Convert>(weights_const, element::f32);
    auto sub_const = op::v0::Constant::create(element::i4, {1}, {1});
    auto sub_convert = std::make_shared<op::v0::Convert>(sub_const, element::f32);
    auto subtract = std::make_shared<op::v1::Subtract>(weights_convert, sub_convert);
    auto mul_const = op::v0::Constant::create(element::f32, {1}, {2.0});
    auto weights = std::make_shared<op::v1::Multiply>(subtract, mul_const);

    auto reshape_weights_pattern =
        op::v0::Constant::create(element::i64, Shape{2}, {weights_shape[0], weights_shape[1]});
    auto reshape_weights = std::make_shared<op::v1::Reshape>(weights, reshape_weights_pattern, false);

    auto input_transpose_const = op::v0::Constant::create(element::i64, Shape{4}, input_transpose_order);
    auto transpose_input = std::make_shared<op::v1::Transpose>(input, input_transpose_const);

    auto matmul = std::make_shared<op::v0::MatMul>(transpose_input, reshape_weights, false, true);

    auto output_transpose_const = op::v0::Constant::create(element::i64, Shape{4}, output_transpose_order);
    auto final_node = std::make_shared<op::v1::Transpose>(matmul, output_transpose_const);

    return std::make_shared<Model>(OutputVector{final_node}, ParameterVector{input});
}
}  // namespace

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_InShape_1_C_1_W) {
    const Shape input_shape{1, 16, 1, 8};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>();
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 2, 3, 1}, {0, 3, 1, 2});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_InShape_N_C_1_1) {
    const Shape input_shape{8, 16, 1, 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>();
    model_ref = create_matmul_function(input_shape, weights_shape, {2, 3, 0, 1}, {2, 3, 0, 1});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_InShape_1_C_H_1) {
    const Shape input_shape{1, 16, 8, 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>();
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 3, 2, 1}, {0, 3, 2, 1});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_Negative_NoDequantize) {
    const Shape input_shape{1, 16, 1, 8};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, false);
    manager.register_pass<pass::ConvertConvolutionToMatMul>();
    model_ref = create_conv_function(input_shape, weights_shape, false);
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_Negative_Not1x1Conv) {
    const Shape input_shape{1, 16, 8, 8};
    const Shape weights_shape{32, 16, 3, 3};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>();
    model_ref = create_conv_function(input_shape, weights_shape, true);
}
