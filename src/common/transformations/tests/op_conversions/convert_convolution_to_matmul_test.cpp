// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_convolution_to_matmul.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
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
std::shared_ptr<Model> create_conv_function(const PartialShape& input_shape,
                                            const Shape& weights_shape,
                                            bool with_decompression,
                                            bool with_zero_point = true) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);

    std::shared_ptr<Node> weights;
    if (with_decompression) {
        weights = ov::test::utils::initMatMulDecompressionSubgraph(
            weights_shape,
            -1,
            element::f32,
            element::i4,
            element::f32,
            element::f32,
            false,
            ov::test::utils::DecompressionType::scalar,
            with_zero_point ? ov::test::utils::DecompressionType::scalar : ov::test::utils::DecompressionType::empty,
            false,
            false);
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

std::shared_ptr<Model> create_matmul_function(const PartialShape& input_shape,
                                              const Shape& weights_shape,
                                              const std::vector<int64_t>& input_transpose_order,
                                              const std::vector<int64_t>& output_transpose_order,
                                              bool with_zero_point = true) {
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);

    auto weights = ov::test::utils::initMatMulDecompressionSubgraph(
        weights_shape,
        -1,
        element::f32,
        element::i4,
        element::f32,
        element::f32,
        false,
        ov::test::utils::DecompressionType::scalar,
        with_zero_point ? ov::test::utils::DecompressionType::scalar : ov::test::utils::DecompressionType::empty,
        false,
        false);

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

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_static_InShape_1_C_1_W) {
    const PartialShape input_shape{1, 16, 1, 8};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 2, 3, 1}, {0, 3, 1, 2});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_static_InShape_N_C_1_1) {
    const PartialShape input_shape{8, 16, 1, 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {2, 3, 0, 1}, {2, 3, 0, 1});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_static_InShape_1_C_H_1) {
    const PartialShape input_shape{1, 16, 8, 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 3, 2, 1}, {0, 3, 2, 1});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_static_InShape_1_C_H_1_without_zp) {
    const PartialShape input_shape{1, 16, 8, 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true, false);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 3, 2, 1}, {0, 3, 2, 1}, false);
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_dynamic_InShape_1_C_1_W) {
    const PartialShape input_shape{1, 16, 1, Dimension::dynamic()};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 2, 3, 1}, {0, 3, 1, 2});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_dynamic_InShape_N_C_1_1) {
    const PartialShape input_shape{Dimension::dynamic(), 16, 1, 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {2, 3, 0, 1}, {2, 3, 0, 1});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_dynamic_InShape_1_C_H_1) {
    const PartialShape input_shape{1, 16, Dimension::dynamic(), 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 3, 2, 1}, {0, 3, 2, 1});
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_dynamic_InShape_1_C_H_1_without_zp) {
    const PartialShape input_shape{1, 16, Dimension::dynamic(), 1};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, true, false);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_matmul_function(input_shape, weights_shape, {0, 3, 2, 1}, {0, 3, 2, 1}, false);
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_Negative_NoDequantize) {
    const PartialShape input_shape{1, 16, 1, 8};
    const Shape weights_shape{32, 16, 1, 1};
    model = create_conv_function(input_shape, weights_shape, false);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_conv_function(input_shape, weights_shape, false);
}

TEST_F(TransformationTestsF, ConvertConvolutionToMatMul_Negative_Not1x1Conv) {
    const PartialShape input_shape{1, 16, 8, 8};
    const Shape weights_shape{32, 16, 3, 3};
    model = create_conv_function(input_shape, weights_shape, true);
    manager.register_pass<pass::ConvertConvolutionToMatMul>(ov::element::TypeVector{element::i4},
                                                            ov::element::TypeVector{});
    model_ref = create_conv_function(input_shape, weights_shape, true);
}
