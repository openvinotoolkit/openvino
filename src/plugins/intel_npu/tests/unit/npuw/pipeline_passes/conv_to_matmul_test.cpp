// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/conv_to_matmul.hpp"

#include <algorithm>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace {

template <class Op>
std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<Op>(op);
    });
}

std::shared_ptr<ov::Model> build_conv_to_matmul_model(bool per_output_channel_scale,
                                                      bool with_output_transpose,
                                                      bool with_input_transpose = true,
                                                      bool use_scale_parameter_chain = false) {
    auto data = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                        with_input_transpose ? ov::Shape{1, 1, 3, 4}
                                                                             : ov::Shape{1, 4, 1, 3});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i8, ov::Shape{2, 4, 1, 1});
    std::shared_ptr<ov::Node> scale_source;
    std::shared_ptr<ov::op::v0::Parameter> scale_parameter;
    if (use_scale_parameter_chain) {
        scale_parameter = std::make_shared<ov::op::v0::Parameter>(ov::element::f32,
                                                                  per_output_channel_scale ? ov::Shape{2} : ov::Shape{8});
        auto scale_reshape_1 = std::make_shared<ov::op::v1::Reshape>(
            scale_parameter,
            ov::op::v0::Constant::create(ov::element::i32,
                                         ov::Shape{1},
                                         std::vector<int32_t>{per_output_channel_scale ? 2 : 8}),
            false);
        scale_source = std::make_shared<ov::op::v1::Reshape>(
            scale_reshape_1,
            ov::op::v0::Constant::create(ov::element::i32,
                                         ov::Shape{4},
                                         per_output_channel_scale ? std::vector<int32_t>{2, 1, 1, 1}
                                                                  : std::vector<int32_t>{2, 4, 1, 1}),
            false);
    } else {
        scale_source = std::make_shared<ov::op::v1::Reshape>(
            ov::op::v0::Constant::create(ov::element::f32,
                                         per_output_channel_scale ? ov::Shape{2} : ov::Shape{8},
                                         per_output_channel_scale ? std::vector<float>{0.25f, 0.5f}
                                                                  : std::vector<float>{0.25f,
                                                                                       0.5f,
                                                                                       0.75f,
                                                                                       1.0f,
                                                                                       1.25f,
                                                                                       1.5f,
                                                                                       1.75f,
                                                                                       2.0f}),
            ov::op::v0::Constant::create(ov::element::i32,
                                         ov::Shape{4},
                                         per_output_channel_scale ? std::vector<int32_t>{2, 1, 1, 1}
                                                                  : std::vector<int32_t>{2, 4, 1, 1}),
            false);
    }

    std::shared_ptr<ov::Node> conv_input = data;
    if (with_input_transpose) {
        conv_input = std::make_shared<ov::op::v1::Transpose>(
            data,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 3, 1, 2}));
    }
    auto converted_weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(converted_weights, scale_source);
    auto convolution = std::make_shared<ov::op::v1::Convolution>(conv_input,
                                                                 scaled_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    convolution->set_friendly_name("conv_to_matmul_out");

    std::shared_ptr<ov::Node> output = convolution;
    if (with_output_transpose) {
        output = std::make_shared<ov::op::v1::Transpose>(
            convolution,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, {0, 2, 3, 1}));
        output->set_friendly_name("conv_to_matmul_output");
    }

    ov::ParameterVector parameters{data, weights};
    if (scale_parameter != nullptr) {
        parameters.push_back(scale_parameter);
    }

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)},
                                       parameters,
                                       "conv_to_matmul_model");
}

std::shared_ptr<ov::Model> build_dumped_gqa_conv_model() {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::Shape{1, 512, 5120});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i4, ov::Shape{5120, 5120, 1, 1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5120});

    auto converted_activation = std::make_shared<ov::op::v0::Convert>(activation, ov::element::f32);
    auto unsqueezed_activation = std::make_shared<ov::op::v0::Unsqueeze>(
        converted_activation,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    auto transposed_activation = std::make_shared<ov::op::v1::Transpose>(
        unsqueezed_activation,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2}));
    auto converted_weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto reshaped_scale = std::make_shared<ov::op::v1::Reshape>(
        scale,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {5120, 1, 1, 1}),
        true);
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(converted_weights, reshaped_scale);
    auto convolution = std::make_shared<ov::op::v1::Convolution>(transposed_activation,
                                                                 scaled_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    auto output = std::make_shared<ov::op::v1::Transpose>(
        convolution,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1}));

    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)},
                                       ov::ParameterVector{activation, weights, scale},
                                       "dumped_gqa_conv_model");
}

TEST(ConvToMatMulPassTest, RewritesPointwiseConvolutionToMatMulWithPostMatMulScale) {
    const auto model = build_conv_to_matmul_model(true, true);

    ov::npuw::ConvToMatMul pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Transpose>(model), 0u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_shape(), ov::Shape({1, 1, 3, 2}));

    const auto result_node = model->get_results().front()->input_value(0).get_node_shared_ptr();
    const auto output_multiply = ov::as_type_ptr<ov::op::v1::Multiply>(result_node);
    ASSERT_NE(output_multiply, nullptr);
    EXPECT_TRUE(ov::is_type<ov::op::v0::MatMul>(output_multiply->input_value(0).get_node_shared_ptr()) ||
                ov::is_type<ov::op::v0::MatMul>(output_multiply->input_value(1).get_node_shared_ptr()));
}

TEST(ConvToMatMulPassTest, KeepsConvolutionWhenScaleIsNotPerOutputChannel) {
    const auto model = build_conv_to_matmul_model(false, true);

    ov::npuw::ConvToMatMul pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 0u);
}

TEST(ConvToMatMulPassTest, RewritesDirectConvolutionToMatMulWithOutputTranspose) {
    const auto model = build_conv_to_matmul_model(true, false, false);

    ov::npuw::ConvToMatMul pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Transpose>(model), 2u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_shape(), ov::Shape({1, 2, 1, 3}));
}

TEST(ConvToMatMulPassTest, RewritesConvolutionWithScaleParameterReshapeChain) {
    const auto model = build_conv_to_matmul_model(true, true, false, true);

    ov::npuw::ConvToMatMul pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(model), 1u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_shape(), ov::Shape({1, 1, 3, 2}));
}

TEST(ConvToMatMulPassTest, RewritesDumpedGQAConvolutionShape) {
    const auto model = build_dumped_gqa_conv_model();

    ov::npuw::ConvToMatMul pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_shape(), ov::Shape({1, 1, 512, 5120}));
}

TEST(ConvToMatMulPassTest, RewritesFrontendWeightsAsInputsConvolutionShape) {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 1, 512, 5120});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i4, ov::Shape{5120, 5120, 1, 1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5120});

    auto transposed_activation = std::make_shared<ov::op::v1::Transpose>(
        activation,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2}));
    auto converted_weights = std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32);
    auto reshaped_scale = std::make_shared<ov::op::v1::Reshape>(
        scale,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {5120, 1, 1, 1}),
        true);
    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(converted_weights, reshaped_scale);
    auto convolution = std::make_shared<ov::op::v1::Convolution>(transposed_activation,
                                                                 scaled_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    auto output = std::make_shared<ov::op::v1::Transpose>(
        convolution,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1}));
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)},
                                                   ov::ParameterVector{activation, weights, scale},
                                                   "frontend_weights_as_inputs_conv_model");

    ov::npuw::ConvToMatMul pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_shape(), ov::Shape({1, 1, 512, 5120}));
}

TEST(ConvToMatMulPassTest, RewritesDumpedMlpDownProjectionShape) {
    auto activation = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 17920, 1, 512});
    auto weights = std::make_shared<ov::op::v0::Parameter>(ov::element::i4, ov::Shape{5120, 17920, 1, 1});
    auto scale = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{5120});

    auto scaled_weights = std::make_shared<ov::op::v1::Multiply>(
        std::make_shared<ov::op::v0::Convert>(weights, ov::element::f32),
        std::make_shared<ov::op::v1::Reshape>(
            scale,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {5120, 1, 1, 1}),
            true));
    auto convolution = std::make_shared<ov::op::v1::Convolution>(activation,
                                                                 scaled_weights,
                                                                 ov::Strides{1, 1},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::CoordinateDiff{0, 0},
                                                                 ov::Strides{1, 1});
    auto output = std::make_shared<ov::op::v1::Transpose>(
        convolution,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1}));
    const auto model = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(output)},
                                                   ov::ParameterVector{activation, weights, scale},
                                                   "dumped_mlp_down_projection_conv_model");

    ov::npuw::ConvToMatMul pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_shape(), ov::Shape({1, 1, 512, 5120}));
}

}  // namespace
