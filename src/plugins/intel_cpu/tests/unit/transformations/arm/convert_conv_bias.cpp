// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/round.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/arm/pass/convert_conv_bias.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace ov::intel_cpu;

static std::shared_ptr<ov::Node> createConvolution(ov::element::Type input_type,
                                                    ov::element::Type weights_type,
                                                    std::shared_ptr<ov::op::v0::Parameter>& input) {
    static const size_t spatial_dims = 2;
    ov::Strides strides(spatial_dims, 1);
    ov::CoordinateDiff pads(spatial_dims, 0);

    input = std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape{1, 3, 224, 224});
    auto weights = ov::op::v0::Constant::create(weights_type, ov::Shape{16, 3, 3, 3}, {1});

    if (input_type == ov::element::f32 && weights_type == ov::element::f32) {
        return std::make_shared<ov::op::v1::Convolution>(
            input, weights, strides, pads, pads, strides);
    } else {
        return std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Convolution>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
            strides, pads, pads, strides);
    }
}

static std::shared_ptr<ov::Node> createAdd(const ov::Output<ov::Node>& input1,
                                            const ov::Output<ov::Node>& input2,
                                            bool force_type_relaxed = false) {
    if (!force_type_relaxed && input1.get_element_type() == input2.get_element_type()) {
        return std::make_shared<ov::op::v1::Add>(input1, input2);
    } else {
        return std::make_shared<ov::op::TypeRelaxed<ov::op::v1::Add>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(input1, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(input2, ov::element::f32).get());
    }
}

// Pattern: Input -> Convolution -> Add(bias) -> Multiply -> Result
static std::shared_ptr<ov::Model> createInitGraph(ov::element::Type input_type,
                                                  ov::element::Type weights_type,
                                                  ov::element::Type bias_type) {
    std::shared_ptr<ov::op::v0::Parameter> input;
    auto conv = createConvolution(input_type, weights_type, input);

    auto bias = ov::op::v0::Constant::create(bias_type, {1, 16, 1, 1}, {1.5f});
    auto add = createAdd(conv, bias);

    auto dequant_scale = ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(add, dequant_scale);

    return std::make_shared<ov::Model>(ov::OutputVector{multiply}, ov::ParameterVector{input});
}

// Pattern: Input -> Convolution -> Add(Round(bias)->Convert(i32)) -> Multiply -> Result
static std::shared_ptr<ov::Model> createRefGraph(ov::element::Type input_type,
                                                 ov::element::Type weights_type,
                                                 ov::element::Type bias_type) {
    std::shared_ptr<ov::op::v0::Parameter> input;
    auto conv = createConvolution(input_type, weights_type, input);

    auto bias = ov::op::v0::Constant::create(bias_type, {1, 16, 1, 1}, {1.5f});
    auto round = std::make_shared<ov::op::v5::Round>(bias, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto convert = std::make_shared<ov::op::v0::Convert>(round, ov::element::i32);

    // The transformation creates a TypeRelaxed Add when the original Add is not TypeRelaxed
    auto add = createAdd(conv, convert, true);

    auto dequant_scale = ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(add, dequant_scale);

    return std::make_shared<ov::Model>(ov::OutputVector{multiply}, ov::ParameterVector{input});
}

// u8 input, i8 weights, f32 bias -> transformation should be applied
TEST_F(TransformationTestsF, ConvertConvolutionBias_U8Input_I8Weights_Applied) {
    model = createInitGraph(ov::element::u8, ov::element::i8, ov::element::f32);
    manager.register_pass<ConvertConvolutionBias>();
    model_ref = createRefGraph(ov::element::u8, ov::element::i8, ov::element::f32);
}

// u8 input, u8 weights, f32 bias -> transformation should be applied
TEST_F(TransformationTestsF, ConvertConvolutionBias_U8Input_U8Weights_Applied) {
    model = createInitGraph(ov::element::u8, ov::element::u8, ov::element::f32);
    manager.register_pass<ConvertConvolutionBias>();
    model_ref = createRefGraph(ov::element::u8, ov::element::u8, ov::element::f32);
}

// i8 input, i8 weights, f32 bias -> transformation should be applied
TEST_F(TransformationTestsF, ConvertConvolutionBias_I8Input_I8Weights_Applied) {
    model = createInitGraph(ov::element::i8, ov::element::i8, ov::element::f32);
    manager.register_pass<ConvertConvolutionBias>();
    model_ref = createRefGraph(ov::element::i8, ov::element::i8, ov::element::f32);
}

// bias already i32 -> transformation should NOT be applied
TEST_F(TransformationTestsF, ConvertConvolutionBias_BiasAlreadyI32_NotApplied) {
    model = createInitGraph(ov::element::u8, ov::element::i8, ov::element::i32);
    manager.register_pass<ConvertConvolutionBias>();
}

// f32 input, f32 weights -> transformation should NOT be applied (not quantized case)
TEST_F(TransformationTestsF, ConvertConvolutionBias_F32Input_F32Weights_NotApplied) {
    model = createInitGraph(ov::element::f32, ov::element::f32, ov::element::f32);
    manager.register_pass<ConvertConvolutionBias>();
}
