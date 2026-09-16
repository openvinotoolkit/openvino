// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npuw_transformations/collapse_unqdq.hpp"
#include "npuw_transformations/conv_to_matmul.hpp"
#include "npuw_transformations/drop_zp_subtract.hpp"

#include <algorithm>

#include <gtest/gtest.h>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace {

template <class Op>
std::size_t count_ops(const std::shared_ptr<ov::Model>& model) {
    const auto ops = model->get_ops();
    return std::count_if(ops.begin(), ops.end(), [](const auto& op) {
        return ov::is_type<Op>(op);
    });
}

std::shared_ptr<ov::Model> build_unqdq_model(const ov::element::Type& input_type = ov::element::f32) {
    auto input = std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape{1, 4});
    auto input_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto input_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {1.0f});
    auto output_low = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {0.0f});
    auto output_high = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {255.0f});
    auto fake_quantize =
        std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 256);
    auto quantized_convert = std::make_shared<ov::op::v0::Convert>(fake_quantize, ov::element::u16);
    auto dequantized_convert = std::make_shared<ov::op::v0::Convert>(quantized_convert, ov::element::f32);
    auto zero_point = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::u16, ov::Shape{}, {128}),
        ov::element::f32);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(dequantized_convert, zero_point);
    auto scale = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::f16, ov::Shape{}, {0.1f}),
        ov::element::f32);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(subtract, scale);
    multiply->set_friendly_name("unqdq_out");
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(multiply)},
                                       ov::ParameterVector{input},
                                       "unqdq_model");
}

std::shared_ptr<ov::Model> build_zero_point_subtract_model(bool zero_point_is_zero) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto zero_point = std::make_shared<ov::op::v0::Convert>(
        ov::op::v0::Constant::create(ov::element::u16, ov::Shape{4}, zero_point_is_zero ? std::vector<uint16_t>{0, 0, 0, 0}
                                                                                         : std::vector<uint16_t>{0, 1, 0, 0}),
        ov::element::f32);
    auto subtract = std::make_shared<ov::op::v1::Subtract>(input, zero_point);
    subtract->set_friendly_name("zp_subtract");
    return std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(subtract)},
                                       ov::ParameterVector{input},
                                       "zero_point_subtract_model");
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
        scale_parameter = std::make_shared<ov::op::v0::Parameter>(
            ov::element::f32,
            per_output_channel_scale ? ov::Shape{2} : ov::Shape{8});
        auto scale_reshape_1 = std::make_shared<ov::op::v1::Reshape>(
            scale_parameter,
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, std::vector<int32_t>{per_output_channel_scale ? 2 : 8}),
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
                                                                  : std::vector<float>{0.25f, 0.5f, 0.75f, 1.0f,
                                                                                       1.25f, 1.5f, 1.75f, 2.0f}),
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

std::shared_ptr<ov::Model> build_conv_to_matmul_and_unqdq_model() {
    const auto conv_model = build_conv_to_matmul_model(true, true);
    const auto unqdq_model = build_unqdq_model();

    ov::ResultVector results;
    for (const auto& result : conv_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }
    for (const auto& result : unqdq_model->get_results()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(result->input_value(0)));
    }

    ov::ParameterVector parameters = conv_model->get_parameters();
    const auto unqdq_parameters = unqdq_model->get_parameters();
    parameters.insert(parameters.end(), unqdq_parameters.begin(), unqdq_parameters.end());

    return std::make_shared<ov::Model>(results, parameters, "conv_to_matmul_and_unqdq_model");
}

void expect_conv_to_matmul_and_unqdq_rewritten(const std::shared_ptr<ov::Model>& model) {
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Convolution>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::MatMul>(model), 1u);
    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
}

TEST(CollapseUNQDQPassTest, DropsEntireUnqdqChainFromModel) {
    const auto model = build_unqdq_model();
    ASSERT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 1u);
    ASSERT_EQ(count_ops<ov::op::v1::Multiply>(model), 1u);

    ov::npuw::CollapseUNQDQ pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::Convert>(model), 0u);

    const auto output_node = model->get_results().front()->input_value(0).get_node_shared_ptr();
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(output_node));
}

TEST(CollapseUNQDQPassTest, PreservesOriginalOutputElementType) {
    const auto model = build_unqdq_model(ov::element::f16);

    ov::npuw::CollapseUNQDQ pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v0::FakeQuantize>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Multiply>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::Convert>(model), 1u);
    EXPECT_EQ(model->get_results().front()->input_value(0).get_element_type(), ov::element::f32);
}

TEST(DropZPSubtractPassTest, DropsSubtractWhenZeroPointIsAllZeros) {
    const auto model = build_zero_point_subtract_model(true);

    ov::npuw::DropZPSubtract pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 0u);
    EXPECT_EQ(count_ops<ov::op::v0::Convert>(model), 0u);
    EXPECT_TRUE(ov::is_type<ov::op::v0::Parameter>(model->get_results().front()->input_value(0).get_node_shared_ptr()));
}

TEST(DropZPSubtractPassTest, KeepsSubtractWhenZeroPointIsNotAllZeros) {
    const auto model = build_zero_point_subtract_model(false);

    ov::npuw::DropZPSubtract pass;
    pass.run_on_model(model);
    model->validate_nodes_and_infer_types();

    EXPECT_EQ(count_ops<ov::op::v1::Subtract>(model), 1u);
}

TEST(CollapseUNQDQPassTest, ConvToMatMulAndUNQDQOrderDoesNotMatter) {
    {
        const auto model = build_conv_to_matmul_and_unqdq_model();
        ov::npuw::ConvToMatMul conv_to_matmul;
        ov::npuw::CollapseUNQDQ collapse_unqdq;

        conv_to_matmul.run_on_model(model);
        collapse_unqdq.run_on_model(model);

        expect_conv_to_matmul_and_unqdq_rewritten(model);
    }

    {
        const auto model = build_conv_to_matmul_and_unqdq_model();
        ov::npuw::ConvToMatMul conv_to_matmul;
        ov::npuw::CollapseUNQDQ collapse_unqdq;

        collapse_unqdq.run_on_model(model);
        conv_to_matmul.run_on_model(model);

        expect_conv_to_matmul_and_unqdq_rewritten(model);
    }
}

}  // namespace
