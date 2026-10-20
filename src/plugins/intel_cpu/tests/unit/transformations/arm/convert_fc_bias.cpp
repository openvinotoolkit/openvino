// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/arm/pass/convert_fc_bias.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/node_builders/fake_quantize.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/round.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/general_utils.h"

using namespace ov::intel_cpu;

static std::shared_ptr<ov::Node> createMatMul(ov::element::Type input_type,
                                              ov::element::Type weights_type,
                                              std::shared_ptr<ov::op::v0::Parameter>& input) {
    input = std::make_shared<ov::op::v0::Parameter>(input_type, ov::Shape{1, 64});
    auto weights = ov::op::v0::Constant::create(weights_type, ov::Shape{64, 16}, {1});

    if (input_type == ov::element::f32 && weights_type == ov::element::f32) {
        return std::make_shared<ov::op::v0::MatMul>(input, weights, false, false);
    } else {
        return std::make_shared<ov::op::TypeRelaxed<ov::op::v0::MatMul>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
            false,
            false);
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

static std::shared_ptr<ov::op::v0::FakeQuantize> createFakeQuantize(const ov::Output<ov::Node>& input,
                                                                    ov::element::Type input_type,
                                                                    size_t levels = 256) {
    const bool is_signed = input_type == ov::element::i8;
    const float low = is_signed ? -128.0f : 0.0f;
    const float high = is_signed ? 127.0f : 255.0f;
    return ov::as_type_ptr<ov::op::v0::FakeQuantize>(
        ov::test::utils::make_fake_quantize(input, input_type, levels, {}, {low}, {high}, {low}, {high}));
}

static std::shared_ptr<ov::op::v0::Constant> createDequantScale(bool per_channel) {
    if (per_channel) {
        std::vector<float> scales(16);
        for (size_t i = 0; i < scales.size(); ++i) {
            scales[i] = 0.25f + 0.05f * static_cast<float>(i);
        }
        return ov::op::v0::Constant::create(ov::element::f32, {1, 16}, scales);
    }
    return ov::op::v0::Constant::create(ov::element::f32, {1}, {0.5f});
}

// Pattern: Input -> MatMul -> Multiply -> Add(bias) -> FQ -> Result
static std::shared_ptr<ov::Model> createInitGraph(ov::element::Type input_type,
                                                  ov::element::Type weights_type,
                                                  ov::element::Type bias_type,
                                                  bool keep_fq_output_precision = false,
                                                  bool per_channel_scale = false) {
    std::shared_ptr<ov::op::v0::Parameter> input;
    auto matmul = createMatMul(input_type, weights_type, input);

    auto dequant_scale = createDequantScale(per_channel_scale);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(matmul, dequant_scale);

    auto bias = ov::op::v0::Constant::create(bias_type, {1, 16}, {1.5f});
    auto add = createAdd(multiply, bias);

    auto fq = createFakeQuantize(add, input_type, 256);
    fq->set_output_type(0, keep_fq_output_precision ? ov::element::f32 : input_type, fq->get_output_shape(0));

    return std::make_shared<ov::Model>(ov::OutputVector{fq}, ov::ParameterVector{input});
}

// Pattern: Input -> MatMul -> Add(Round(bias)->Convert(i32)) -> Multiply -> FQ -> Result
static std::shared_ptr<ov::Model> createRefGraph(ov::element::Type input_type,
                                                 ov::element::Type weights_type,
                                                 ov::element::Type bias_type,
                                                 bool per_channel_scale = false) {
    std::shared_ptr<ov::op::v0::Parameter> input;
    auto matmul = createMatMul(input_type, weights_type, input);

    auto bias = ov::op::v0::Constant::create(bias_type, {1, 16}, {1.5f});
    auto round = std::make_shared<ov::op::v5::Round>(bias, ov::op::v5::Round::RoundMode::HALF_TO_EVEN);
    auto convert = std::make_shared<ov::op::v0::Convert>(round, ov::element::i32);

    auto add = createAdd(matmul, convert, true);
    auto dequant_scale = createDequantScale(per_channel_scale);
    auto multiply = std::make_shared<ov::op::v1::Multiply>(add, dequant_scale);
    ov::mark_as_dequantization_node(multiply);

    auto fq = createFakeQuantize(multiply, input_type, 256);

    return std::make_shared<ov::Model>(ov::OutputVector{fq}, ov::ParameterVector{input});
}

// u8 act, i8 weights, f32 bias -> transformation should be applied
TEST_F(TransformationTestsF, ConvertFullyConnectedBias_U8Act_I8Weights_Applied) {
    model = createInitGraph(ov::element::u8, ov::element::i8, ov::element::f32);
    manager.register_pass<ConvertFullyConnectedBias>();
    model_ref = createRefGraph(ov::element::u8, ov::element::i8, ov::element::f32);
}

// i8 act, i8 weights, f32 bias -> transformation should be applied
TEST_F(TransformationTestsF, ConvertFullyConnectedBias_I8Act_I8Weights_Applied) {
    model = createInitGraph(ov::element::i8, ov::element::i8, ov::element::f32);
    manager.register_pass<ConvertFullyConnectedBias>();
    model_ref = createRefGraph(ov::element::i8, ov::element::i8, ov::element::f32);
}

// i8 act, i8 weights, f32 bias, PER-CHANNEL dequantization scale -> transformation should be applied
TEST_F(TransformationTestsF, ConvertFullyConnectedBias_I8Act_I8Weights_PerChannelScale_Applied) {
    model = createInitGraph(ov::element::i8, ov::element::i8, ov::element::f32, false, true);
    manager.register_pass<ConvertFullyConnectedBias>();
    model_ref = createRefGraph(ov::element::i8, ov::element::i8, ov::element::f32, true);
}

// u8 act, u8 weights, f32 bias -> transformation should NOT be applied.
TEST_F(TransformationTestsF, ConvertFullyConnectedBias_U8Act_U8Weights_NotApplied) {
    model = createInitGraph(ov::element::u8, ov::element::u8, ov::element::f32);
    manager.register_pass<ConvertFullyConnectedBias>();
}

// bias already i32 -> transformation should NOT be applied
TEST_F(TransformationTestsF, ConvertFullyConnectedBias_BiasAlreadyI32_NotApplied) {
    model = createInitGraph(ov::element::u8, ov::element::i8, ov::element::i32);
    manager.register_pass<ConvertFullyConnectedBias>();
}

// f32 act, f32 weights -> transformation should NOT be applied (not quantized case)
TEST_F(TransformationTestsF, ConvertFullyConnectedBias_F32Act_F32Weights_NotApplied) {
    model = createInitGraph(ov::element::f32, ov::element::f32, ov::element::f32);
    manager.register_pass<ConvertFullyConnectedBias>();
}

// fq output does not match activation precision -> transformation should NOT be applied
TEST_F(TransformationTestsF, ConvertFullyConnectedBias_FqOutputDoesNotMatchActivationPrecision_NotApplied) {
    model = createInitGraph(ov::element::i8, ov::element::i8, ov::element::f32, true);
    manager.register_pass<ConvertFullyConnectedBias>();
}
