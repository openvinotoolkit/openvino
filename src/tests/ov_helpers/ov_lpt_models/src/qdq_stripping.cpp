// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/qdq_stripping.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/type_ranges.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"

namespace ov {
namespace builder {
namespace subgraph {

// QuantizationParams implementation
ov::Output<ov::Node> QuantizationParams::build_fq(const ov::Output<ov::Node>& input) const {
    auto input_low = ov::op::v0::Constant::create(ov::element::f32, {}, {i_l});
    auto input_high = ov::op::v0::Constant::create(ov::element::f32, {}, {i_h});
    auto output_low = ov::op::v0::Constant::create(ov::element::f32, {}, {o_l});
    auto output_high = ov::op::v0::Constant::create(ov::element::f32, {}, {o_h});
    return std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 65536);
}

ov::Output<ov::Node> QuantizationParams::build_dq(const ov::Output<ov::Node>& input,
                                                   const ov::element::Type& quantization_precision) const {
    auto act_zero_point = ov::op::v0::Constant::create(quantization_precision, {}, {zero_point});
    auto act_zp_convert = std::make_shared<ov::op::v0::Convert>(act_zero_point, ov::element::f32);

    auto act_subtract = std::make_shared<ov::op::v1::Subtract>(input, act_zp_convert);
    float scale_value = (i_h - i_l) / (o_h - o_l);
    auto act_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value});

    return std::make_shared<ov::op::v1::Multiply>(act_subtract, act_scale);
}

ov::Output<ov::Node> QDQStrippingFunction::build_dq_subgraph(ov::element::Type quantized_type,
                                                              const ov::Shape& shape,
                                                              float scale_value,
                                                              int zero_point,
                                                              std::optional<size_t> seed,
                                                              std::optional<std::vector<int>> constant_values,
                                                              float constant_value) {
    std::shared_ptr<ov::Node> quantized_const;

    if (seed.has_value()) {
        auto gen_data = ov::test::utils::rangeByType.get_range(quantized_type);
        gen_data.seed = seed.value();
        quantized_const = ov::test::utils::make_constant(quantized_type, shape, gen_data);
    } else if (constant_values.has_value()) {
        quantized_const = ov::test::utils::make_constant(quantized_type, shape, constant_values.value());
    } else {
        quantized_const = ov::op::v0::Constant::create(quantized_type, shape, {constant_value});
    }

    auto convert = std::make_shared<ov::op::v0::Convert>(quantized_const, ov::element::f32);

    std::shared_ptr<ov::Node> result = convert;
    if (zero_point != 0) {
        auto zp_quantized = ov::op::v0::Constant::create(quantized_type, {}, {zero_point});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_quantized, ov::element::f32);
        result = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
    }

    auto scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value});
    result = std::make_shared<ov::op::v1::Multiply>(result, scale);

    return result;
}

ov::Output<ov::Node> QDQStrippingFunction::add_bias(const ov::Output<ov::Node>& conv,
                                                     const ov::Output<ov::Node>& bias) {
    const auto conv_shape = std::make_shared<ov::op::v3::ShapeOf>(conv);
    const auto conv_rank = std::make_shared<ov::op::v3::ShapeOf>(conv_shape);

    const auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    const auto two_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
    const auto tail_shape_rank = std::make_shared<ov::op::v1::Subtract>(conv_rank, two_const);
    const auto tail_shape = std::make_shared<ov::op::v3::Broadcast>(one_const, tail_shape_rank);

    const auto C_dim = std::make_shared<ov::op::v3::ShapeOf>(bias);
    const auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one_const, C_dim, tail_shape}, 0);
    const auto reshaped_bias = std::make_shared<ov::op::v1::Reshape>(bias, new_shape, false);

    return std::make_shared<ov::op::v1::Add>(conv, reshaped_bias);
}

ov::Output<ov::Node> QDQStrippingFunction::build_fq(const ov::Output<ov::Node>& input,
                                                     float input_low, float input_high,
                                                     float output_low, float output_high,
                                                     size_t levels) {
    auto il = ov::op::v0::Constant::create(ov::element::f32, {}, {input_low});
    auto ih = ov::op::v0::Constant::create(ov::element::f32, {}, {input_high});
    auto ol = ov::op::v0::Constant::create(ov::element::f32, {}, {output_low});
    auto oh = ov::op::v0::Constant::create(ov::element::f32, {}, {output_high});
    return std::make_shared<ov::op::v0::FakeQuantize>(input, il, ih, ol, oh, levels);
}

ov::Output<ov::Node> QDQStrippingFunction::build_dq(const ov::Output<ov::Node>& input,
                                                     const ov::element::Type& quantization_precision,
                                                     float input_low, float input_high,
                                                     float output_low, float output_high,
                                                     int zero_point) {
    auto act_zero_point = ov::op::v0::Constant::create(quantization_precision, {}, {zero_point});
    auto act_zp_convert = std::make_shared<ov::op::v0::Convert>(act_zero_point, ov::element::f32);
    auto act_subtract = std::make_shared<ov::op::v1::Subtract>(input, act_zp_convert);

    float scale_value = (input_high - input_low) / (output_high - output_low);
    auto act_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value});
    return std::make_shared<ov::op::v1::Multiply>(act_subtract, act_scale);
}

// ==============================================================================
// SharedDQ pattern: two Conv branches sharing a quantized input
// FQ y_scale < 1 → stripped without scale propagation
// ==============================================================================

std::shared_ptr<ov::Model> QDQStrippingFunction::getOriginalSharedDQ(const ov::PartialShape& input_shape) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // FQ with y_scale = 10/65535 ≈ 0.00015 < 1 → no scale propagation
    // After FQ+DQ fusion, input and output ranges are the same
    auto input_fq = build_fq(params[0], 0.f, 10.f, 0.f, 10.f);
    auto input_convert1 = std::make_shared<ov::op::v0::Convert>(input_fq, ov::element::u16);
    auto input_convert2 = std::make_shared<ov::op::v0::Convert>(input_convert1, ov::element::f32);

    // scale_value = (i_h - i_l) / (o_h - o_l) = 10 / 65535
    float dq_scale = 10.f / 65535.f;
    auto create_branch = [&](float weight_scale) {
        auto input_dq = build_dq(input_convert2, ov::element::u16, 0.f, 10.f, 0.f, 65535.f, 0);

        auto weight = build_dq_subgraph(ov::element::i8, {4, 3, 1, 1}, weight_scale);
        auto conv = std::make_shared<ov::op::v1::Convolution>(input_dq, weight,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::Strides{1, 1});

        // Second FQ with y_scale < 1 → also stripped without propagation
        // After FQ+DQ fusion, input and output ranges are the same
        auto fq2 = build_fq(conv, -5.f, 5.f, -5.f, 5.f);
        auto conv_convert1 = std::make_shared<ov::op::v0::Convert>(fq2, ov::element::i16);
        auto conv_convert2 = std::make_shared<ov::op::v0::Convert>(conv_convert1, ov::element::f32);
        return build_dq(conv_convert2, ov::element::i16, -5.f, 5.f, -32768.f, 32767.f, 0);
    };

    auto left = create_branch(0.01f);
    auto right = create_branch(0.02f);
    auto add = std::make_shared<ov::op::v1::Add>(left, right);

    return std::make_shared<ov::Model>(ov::OutputVector{add}, params, "SharedDQ");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::getReferenceSharedDQ(const ov::PartialShape& input_shape) {
    // Reference: same graph with all FQs removed (y_scale < 1, no weight scaling needed)
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // FQ is stripped → its input passes through directly
    auto input_convert1 = std::make_shared<ov::op::v0::Convert>(params[0], ov::element::u16);
    auto input_convert2 = std::make_shared<ov::op::v0::Convert>(input_convert1, ov::element::f32);

    auto create_branch = [&](float weight_scale) {
        auto input_dq = build_dq(input_convert2, ov::element::u16, 0.f, 10.f, 0.f, 65535.f, 0);

        auto weight = build_dq_subgraph(ov::element::i8, {4, 3, 1, 1}, weight_scale);
        auto conv = std::make_shared<ov::op::v1::Convolution>(input_dq, weight,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::Strides{1, 1});

        // Second FQ stripped → Conv output passes through directly
        auto conv_convert1 = std::make_shared<ov::op::v0::Convert>(conv, ov::element::i16);
        auto conv_convert2 = std::make_shared<ov::op::v0::Convert>(conv_convert1, ov::element::f32);
        return build_dq(conv_convert2, ov::element::i16, -5.f, 5.f, -32768.f, 32767.f, 0);
    };

    auto left = create_branch(0.01f);
    auto right = create_branch(0.02f);
    auto add = std::make_shared<ov::op::v1::Add>(left, right);

    return std::make_shared<ov::Model>(ov::OutputVector{add}, params, "SharedDQ");
}

// ==============================================================================
// NeedScalingMulMatMul: two params multiplied by shared DQ constant, MatMul, FQ→DQ
// FQ y_scale = 2 → weights divided by 2
// ==============================================================================

std::shared_ptr<ov::Model> QDQStrippingFunction::getOriginalNeedScalingMulMatMul(const ov::PartialShape& input_shape) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    ov::ParameterVector params{param1, param2};

    // Shared DQ constant: value=10, scale=1.0, zp=0 → DQ = 10 * 1.0 = 10
    auto common_constant = build_dq_subgraph(ov::element::i8, {}, 1.0f, 0, std::nullopt, std::nullopt, 10.f);

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(param1, common_constant);
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(param2, common_constant);

    auto softmax_mul1 = std::make_shared<ov::op::v8::Softmax>(mul1, 1);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(mul1, mul2, false, true);

    // FQ: y_scale = (input_high - input_low) / (levels - 1) = 131070 / 65535 = 2
    // After FQ+DQ fusion, input and output ranges are the same
    auto fq = build_fq(matmul, 0.f, 131070.f, 0.f, 131070.f);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, ov::element::u16);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, ov::element::u16, 0.f, 131070.f, 0.f, 65535.f, 0);

    auto softmax = std::make_shared<ov::op::v8::Softmax>(dq, 1);

    return std::make_shared<ov::Model>(ov::OutputVector{softmax_mul1, softmax}, params, "NeedScalingMulMatMul");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::getReferenceNeedScalingMulMatMul(const ov::PartialShape& input_shape) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    ov::ParameterVector params{param1, param2};

    // Reference: DQ constant scale divided by y_scale=2 → scale becomes 1.0/2 = 0.5
    // The transformation uses make_try_fold which folds Divide(Constant, Constant) to Constant.
    auto common_const = ov::op::v0::Constant::create(ov::element::i8, {}, {10.f});
    auto convert = std::make_shared<ov::op::v0::Convert>(common_const, ov::element::f32);
    auto new_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {0.5f});  // 1.0 / 2.0
    auto common_constant = std::make_shared<ov::op::v1::Multiply>(convert, new_scale);

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(param1, common_constant);
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(param2, common_constant);

    auto softmax_mul1 = std::make_shared<ov::op::v8::Softmax>(mul1, 1);

    // FQ stripped → matmul output passes through directly
    auto matmul = std::make_shared<ov::op::v0::MatMul>(mul1, mul2, false, true);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(matmul, ov::element::u16);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, ov::element::u16, 0.f, 131070.f, 0.f, 65535.f, 0);

    auto softmax = std::make_shared<ov::op::v8::Softmax>(dq, 1);

    return std::make_shared<ov::Model>(ov::OutputVector{softmax_mul1, softmax}, params, "NeedScalingMulMatMul");
}

// ==============================================================================
// NeedScalingMatMulWithBias: MatMul + bias + FQ→DQ→MVN
// FQ y_scale = 4 → both weights and bias divided by 4
// ==============================================================================

std::shared_ptr<ov::Model> QDQStrippingFunction::getOriginalNeedScalingMatMulWithBias(const ov::PartialShape& input_shape) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // Weight DQ: value=1, scale=0.5, zp=0 → DQ = 1 * 0.5 = 0.5
    auto weight = build_dq_subgraph(ov::element::i8, {3, 4}, 0.5f);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], weight, false, false);

    // Bias DQ: value=5, scale=2.0, zp=0 → DQ = 5 * 2.0 = 10
    auto bias = build_dq_subgraph(ov::element::i8, {4}, 2.0f, 0, std::nullopt, std::nullopt, 5.f);
    auto matmul_biased = std::make_shared<ov::op::v1::Add>(matmul, bias);

    // FQ: y_scale = 262140 / 65535 = 4
    // After FQ+DQ fusion, input and output ranges are the same
    auto fq = build_fq(matmul_biased, 0.f, 262140.f, 0.f, 262140.f);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, ov::element::u16);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, ov::element::u16, 0.f, 262140.f, 0.f, 65535.f, 0);

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mvn = std::make_shared<ov::op::v6::MVN>(dq, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "NeedScalingMatMulWithBias");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::getReferenceNeedScalingMatMulWithBias(const ov::PartialShape& input_shape) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // Reference: weight scale divided by 4 → scale = 0.5/4 = 0.125
    // The transformation uses make_try_fold which folds Divide(Constant, Constant) to Constant.
    auto w_const = ov::op::v0::Constant::create(ov::element::i8, {3, 4}, {1.f});
    auto w_convert = std::make_shared<ov::op::v0::Convert>(w_const, ov::element::f32);
    auto w_new_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {0.125f});  // 0.5 / 4.0
    auto weight = std::make_shared<ov::op::v1::Multiply>(w_convert, w_new_scale);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], weight, false, false);

    // Reference: bias scale divided by 4 → scale = 2.0/4 = 0.5
    auto b_const = ov::op::v0::Constant::create(ov::element::i8, {4}, {5.f});
    auto b_convert = std::make_shared<ov::op::v0::Convert>(b_const, ov::element::f32);
    auto b_new_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {0.5f});  // 2.0 / 4.0
    auto bias = std::make_shared<ov::op::v1::Multiply>(b_convert, b_new_scale);

    auto matmul_biased = std::make_shared<ov::op::v1::Add>(matmul, bias);

    // FQ stripped
    auto convert1 = std::make_shared<ov::op::v0::Convert>(matmul_biased, ov::element::u16);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, ov::element::u16, 0.f, 262140.f, 0.f, 65535.f, 0);

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mvn = std::make_shared<ov::op::v6::MVN>(dq, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "NeedScalingMatMulWithBias");
}

// ==============================================================================
// NeedScalingResidualBlock: Conv→bias→FQ→DQ→FQ(fwd)→residual blocks→MVN
// First FQ y_scale=10, forward-path FQ and branch FQs adjusted then stripped.
// ==============================================================================

// Internal helper: builds a dq subgraph where the Multiply scale constant has already
// been divided by a divisor (pre-computed, matching how make_try_fold folds
// Divide(Constant, Constant) to a Constant in the transformation).
static ov::Output<ov::Node> build_scaled_dq_subgraph(ov::element::Type quantized_type,
                                                      const ov::Shape& shape,
                                                      float scale_value,
                                                      float divisor,
                                                      int zero_point,
                                                      float constant_value) {
    auto quantized_const = ov::op::v0::Constant::create(quantized_type, shape, {constant_value});
    auto convert = std::make_shared<ov::op::v0::Convert>(quantized_const, ov::element::f32);

    std::shared_ptr<ov::Node> result = convert;
    if (zero_point != 0) {
        auto zp_quantized = ov::op::v0::Constant::create(quantized_type, {}, {zero_point});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_quantized, ov::element::f32);
        result = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);
    }

    auto folded_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value / divisor});
    result = std::make_shared<ov::op::v1::Multiply>(result, folded_scale);

    return result;
}

std::shared_ptr<ov::Model> QDQStrippingFunction::getOriginalNeedScalingResidualBlock(const ov::PartialShape& input_shape) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // Conv1 weights: value=1, scale=0.5, zp=0 → DQ = 0.5
    auto weight1 = build_dq_subgraph(ov::element::i8, {4, 3, 1, 1}, 0.5f);
    auto conv1 = std::make_shared<ov::op::v1::Convolution>(params[0], weight1,
                                                            ov::Strides{1, 1},
                                                            ov::CoordinateDiff{0, 0},
                                                            ov::CoordinateDiff{0, 0},
                                                            ov::Strides{1, 1});

    // Bias1: value=1, scale=0.1, zp=0 → DQ = 0.1
    auto bias1 = build_dq_subgraph(ov::element::i8, {4}, 0.1f);
    auto conv1_biased = add_bias(conv1, bias1);

    // First FQ: y_scale = (0 - (-655350)) / 65535 = 655350/65535 = 10
    // After FQ+DQ fusion, input and output ranges are the same
    auto fq = build_fq(conv1_biased, -655350.f, 0.f, -655350.f, 0.f);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, ov::element::i16);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, ov::element::i16, -655350.f, 0.f, -65535.f, 0.f, 0);

    // Forward-path FQ: y_scale = 655350/65535 = 10, after /10 → y_scale = 1.0 → stripped
    auto fq_fwd = build_fq(dq, -655350.f, 0.f, -655350.f, 0.f);

    // Branch FQ range
    float branch_lo = -65600.f;
    float branch_hi = 0.f;

    // Residual blocks
    auto create_residual_block = [&](const ov::Output<ov::Node>& input) {
        auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
        auto mvn = std::make_shared<ov::op::v6::MVN>(input, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

        auto weight = build_dq_subgraph(ov::element::i8, {4, 4, 1, 1}, 0.5f);
        auto conv = std::make_shared<ov::op::v1::Convolution>(mvn, weight,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::Strides{1, 1});

        auto bias = build_dq_subgraph(ov::element::i8, {4}, 0.1f);
        auto conv_biased = add_bias(conv, bias);

        // Branch FQ: y_scale = 65600/65535 ≈ 1.001, after /10 → ≈ 0.1 → stripped
        auto bfq = build_fq(conv_biased, branch_lo, branch_hi, branch_lo, branch_hi);

        return std::make_shared<ov::op::v1::Add>(bfq, input);
    };

    auto add1 = create_residual_block(fq_fwd);
    auto add2 = create_residual_block(add1);
    auto add3 = create_residual_block(add2);

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
    auto final_mvn = std::make_shared<ov::op::v6::MVN>(add3, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{final_mvn}, params, "NeedScalingResidualBlock");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::getReferenceNeedScalingResidualBlock(const ov::PartialShape& input_shape) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    const float scale_divisor = 10.f;

    // Conv1: weight scale divided by 10 → scale = 0.5/10 = 0.05
    auto weight1 = build_scaled_dq_subgraph(ov::element::i8, {4, 3, 1, 1}, 0.5f, scale_divisor, 0, 1.f);
    auto conv1 = std::make_shared<ov::op::v1::Convolution>(params[0], weight1,
                                                            ov::Strides{1, 1},
                                                            ov::CoordinateDiff{0, 0},
                                                            ov::CoordinateDiff{0, 0},
                                                            ov::Strides{1, 1});

    // Bias1: scale divided by 10 → scale = 0.1/10 = 0.01
    auto bias1 = build_scaled_dq_subgraph(ov::element::i8, {4}, 0.1f, scale_divisor, 0, 1.f);
    auto conv1_biased = add_bias(conv1, bias1);

    // First FQ stripped
    auto convert1 = std::make_shared<ov::op::v0::Convert>(conv1_biased, ov::element::i16);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, ov::element::i16, -655350.f, 0.f, -65535.f, 0.f, 0);

    // Forward-path FQ stripped (adjusted range: -655350/10 = -65535, 0)
    // Note: its input just passes through

    // Residual blocks with all FQs stripped and weights divided by 10
    auto create_residual_block = [&](const ov::Output<ov::Node>& input) {
        auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
        auto mvn = std::make_shared<ov::op::v6::MVN>(input, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

        // Weight scale divided by 10
        auto weight = build_scaled_dq_subgraph(ov::element::i8, {4, 4, 1, 1}, 0.5f, scale_divisor, 0, 1.f);
        auto conv = std::make_shared<ov::op::v1::Convolution>(mvn, weight,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::CoordinateDiff{0, 0},
                                                               ov::Strides{1, 1});

        // Bias scale divided by 10
        auto bias = build_scaled_dq_subgraph(ov::element::i8, {4}, 0.1f, scale_divisor, 0, 1.f);
        auto conv_biased = add_bias(conv, bias);

        // Branch FQ stripped
        return std::make_shared<ov::op::v1::Add>(conv_biased, input);
    };

    auto add1 = create_residual_block(dq);
    auto add2 = create_residual_block(add1);
    auto add3 = create_residual_block(add2);

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
    auto final_mvn = std::make_shared<ov::op::v6::MVN>(add3, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{final_mvn}, params, "NeedScalingResidualBlock");
}

// ==============================================================================
// GPU accuracy test model builders (shared between LPT unit tests and GPU func tests)
// These use per-precision QuantizationParams maps with i16 ranges chosen for exact
// float32 arithmetic in ConvertQuantizeDequantize fusion (il = -32768*K, ih = 32767*K).
// ==============================================================================

std::shared_ptr<ov::Model> QDQStrippingFunction::build_shared_dq_pattern(const ov::PartialShape& input_shape,
                                                                          const ov::element::Type& quantization_precision) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};
    static const std::unordered_map<ov::element::Type_t, std::pair<QuantizationParams, QuantizationParams>> quantization_params{
        {ov::element::Type_t::u16, {{0.f, 10.f, 0.f, 65535.f, 0}, {-624.4578838348389f, 634.7373962402344f, 0.f, 65535.f, 32500}}},
        {ov::element::Type_t::i16,
         {{-5.0000762939453125f, 4.9999237060546875f, -32768.f, 32767.f, 0}, {-630.0096435546875f, 629.9904174804688f, -32768.f, 32767.f, 0}}},
    };

    const auto& q_params = quantization_params.at(quantization_precision);
    const auto& qp_1 = q_params.first;
    auto input_fq = qp_1.build_fq(params[0]);

    auto input_convert1 = std::make_shared<ov::op::v0::Convert>(input_fq, quantization_precision);
    auto input_convert2 = std::make_shared<ov::op::v0::Convert>(input_convert1, ov::element::f32);

    size_t seed = 1;
    auto create_qdq_branch = [&](float weight_scale_value) {
        auto input_dequantized = qp_1.build_dq(input_convert2, quantization_precision);
        ov::test::utils::InputGenerateData weights_gen_data;
        weights_gen_data.seed = seed;
        auto weight_quantized = ov::test::utils::make_constant(ov::element::i8, ov::Shape{32, 3, 3, 3}, weights_gen_data);
        auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight_quantized, ov::element::f32);
        auto weight_scale = ov::test::utils::make_constant(ov::element::f32, {}, std::vector<float>{weight_scale_value});
        auto weight_dequantized = std::make_shared<ov::op::v1::Multiply>(weight_convert, weight_scale);

        auto conv = std::make_shared<ov::op::v1::Convolution>(input_dequantized,
                                                              weight_dequantized,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});

        ov::test::utils::InputGenerateData bias_gen_data(-2.0, 4, 100, seed++);
        auto bias_const = ov::test::utils::make_constant(ov::element::f32, ov::Shape{1, 32, 1, 1}, bias_gen_data);
        auto conv_biased = std::make_shared<ov::op::v1::Add>(conv, bias_const);

        const auto& qp_2 = q_params.second;
        auto fake_quantize = qp_2.build_fq(conv_biased);
        auto act_quantized = std::make_shared<ov::op::v0::Convert>(fake_quantize, quantization_precision);
        auto act_convert = std::make_shared<ov::op::v0::Convert>(act_quantized, ov::element::f32);
        return qp_2.build_dq(act_convert, quantization_precision);
    };

    auto left_branch = create_qdq_branch(1e-4f);
    auto right_branch = create_qdq_branch(1e-5f);
    auto add_branches = std::make_shared<ov::op::v1::Add>(left_branch, right_branch);

    return std::make_shared<ov::Model>(ov::OutputVector{add_branches}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_need_scaling_mul_matmul_pattern(const ov::PartialShape& input_shape,
                                                                                        const ov::element::Type& quantization_precision) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    ov::ParameterVector params{param1, param2};

    // Weight DQ pattern: quantized constant -> convert -> subtract (zero point) -> multiply (scale)
    // DQ = (127 - 0) * 1.0 = 127. Large constant ensures overflow without scale adjustment:
    // input (up to 1000) * 127 = 127,000 >> f16 max (65504)
    auto common_constant = build_dq_subgraph(ov::element::i8, {}, 1.0f, 0, std::nullopt, std::vector<int>{127});

    // param1 * common_constant
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(param1, common_constant);

    // param2 * common_constant
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(param2, common_constant);

    // Softmax on mul1 detects overflow: if mul1 contains inf, Softmax produces NaN.
    // Axis=1 (dim=3) avoids f16 precision argmax ties that occur with large dim.
    auto softmax_mul1 = std::make_shared<ov::op::v8::Softmax>(mul1, 1);

    // MatMul
    auto matmul = std::make_shared<ov::op::v0::MatMul>(mul1, mul2, false, true);

    // y_scale = (input_high - input_low) / (levels - 1) ≈ 131070 / 65535 = 2
    // Without scale adjustment: DQ = 127, mul1 = input * 127 → up to 127,000 → overflows f16 → NaN
    // With scale adjustment: DQ / y_scale = 127 / 2 = 63.5,
    //   mul1 = input * 63.5 → up to 63,500 → fits f16.
    //   Small FQ step (=2) preserves Softmax input differences accurately.
    //   Softmax axis=1 (3 channels): channel gaps >> f16 step → stable argmax.
    static const std::unordered_map<ov::element::Type_t, QuantizationParams> quantization_params{
        {ov::element::Type_t::u16, {0.f, 131070.f, 0.f, 65535.f, 0}},
        {ov::element::Type_t::i16, {-65536.f, 65534.f, -32768.f, 32767.f, 0}},
    };

    const auto& qp = quantization_params.at(quantization_precision);
    auto fq = qp.build_fq(matmul);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = qp.build_dq(convert2, quantization_precision);

    // Axis=1 (dim=3): Softmax over 3 elements avoids argmax ties from f16 rounding.
    auto softmax = std::make_shared<ov::op::v8::Softmax>(dq, 1);
    return std::make_shared<ov::Model>(ov::OutputVector{softmax_mul1, softmax}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_need_scaling_matmul_with_bias_pattern(const ov::PartialShape& input_shape,
                                                                                              const ov::element::Type& quantization_precision) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // Weight DQ for MatMul: input last dim = 128 (from input shape [1,3,128,128]), output = 32
    // Zero-point -128 shifts i8[-128,127] to [0,255]. Scale=0.02 → weights in [0, 5.1].
    // All-positive weights + all-positive inputs [0,100] → MatMul output always positive.
    // This avoids u16 FQ clamping (input_low=0) which would cause large f16 rounding errors.
    // With 128-element dot product: avg output ≈ 50 * 2.55 * 128 ≈ 16320.
    auto weight = build_dq_subgraph(ov::element::i8, ov::Shape{128, 32}, 0.02f, -128, 1);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(params[0], weight, false, false);

    // Bias DQ: [32] with zero_point=-128 so DQ values are always non-negative.
    // DQ = (i8_val + 128) * 200 → [0, 51000], avg ~25500.
    // Bias is significant relative to MatMul output (~16320) so MVN detects unscaled bias.
    auto bias = build_dq_subgraph(ov::element::i8, {32}, 200.0f, -128, 2);
    auto matmul_biased = std::make_shared<ov::op::v1::Add>(matmul, bias);

    // y_scale = 262140 / 65535 = 4
    // All values are positive (positive weights, positive inputs, positive bias).
    // Total = MatMul(~16320) + bias(max 51000) = max ~67320 → overflows f16 (65504).
    // With scale adjustment (÷4): MatMul ~4080, bias max ~12750, total ~16830 → fits f16.
    // Without scale adjustment: total up to ~67320 → overflows f16 → inf → MVN = NaN.
    // FQ step=4 vs signal ~30000 → 0.01% error → MVN output accurate.
    static const std::unordered_map<ov::element::Type_t, QuantizationParams> quantization_params{
        {ov::element::Type_t::u16, {0.f, 262140.f, 0.f, 65535.f, 0}},
        {ov::element::Type_t::i16, {-131072.f, 131068.f, -32768.f, 32767.f, 0}},
    };

    const auto& qp = quantization_params.at(quantization_precision);
    auto fq = qp.build_fq(matmul_biased);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = qp.build_dq(convert2, quantization_precision);

    // MVN is scale-invariant (normalizes by mean/variance), triggering scale adjustment.
    // Unlike Softmax (which is exponentially sensitive to input perturbations),
    // MVN output is linearly affected by quantization error, allowing tight accuracy checks.
    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mvn = std::make_shared<ov::op::v6::MVN>(dq, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_need_scaling_residual_block_pattern(const ov::PartialShape& input_shape,
                                                                                            const ov::element::Type& quantization_precision) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // First convolution with weight DQ
    // zp=-128 shifts i8 range to [0, 255] → all-positive DQ weights,
    // avoiding u16 FQ clamping at input_low=0 which causes large errors with mixed-sign weights.
    // scale=0.003 produces DQ values up to ~0.765, large enough to cause f16 overflow
    // after y_scale=10 FQ, validating the scale adjustment logic.
    auto weight1 = build_dq_subgraph(ov::element::i8, ov::Shape{32, 3, 3, 3}, 0.003f, -128, 1);
    auto conv1 = std::make_shared<ov::op::v1::Convolution>(params[0],
                                                           weight1,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::Strides{1, 1});

    // Bias with DQ for first convolution (1D bias: [32])
    auto bias1 = build_dq_subgraph(ov::element::i8, {32}, 0.001f, 0);
    auto conv1_biased = add_bias(conv1, bias1);

    // y_scale = (input_high - input_low) / (levels - 1) ≈ 655350 / 65535 ≈ 10
    // After DQ: quantized_value * 10 can reach ~655350, far beyond f16 max (65504)
    // Without scale adjustment: Softmax receives inf -> exp(inf) = inf -> inf/inf = NaN
    // With scale adjustment: weights are divided by ~10, keeping values in f16 range
    static const std::unordered_map<ov::element::Type_t, QuantizationParams> quantization_params{
        {ov::element::Type_t::u16, {0.f, 655350.f, 0.f, 65535.f, 0}},
        {ov::element::Type_t::i16, {-327680.f, 327670.f, -32768.f, 32767.f, 0}},
    };

    const auto& qp = quantization_params.at(quantization_precision);
    auto fq = qp.build_fq(conv1_biased);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = qp.build_dq(convert2, quantization_precision);

    // Insert FQs with 65536 levels (same as levels_to_strip) on both the forward
    // propagation path and the Conv+bias branches of each residual block.
    // These FQs test the full adjust+strip pipeline:
    //   1. First FQ (y_scale=10) is stripped → forward propagation adjusts downstream FQ ranges by /10
    //   2. Forward propagation also triggers backward propagation at each Add, adjusting branch FQ ranges
    //   3. When the topological walk reaches each adjusted FQ, y_scale ≤ 1 → stripped without propagation
    //
    // Forward-path FQ range matches the DQ output range so it doesn't clamp in the
    // reference model. y_scale = range/65535 ≈ 10, after /10 ≈ 1.0 ≤ threshold.
    // Branch FQ range is [0, 65600] / [-32800, 32800] (y_scale ≈ 1.001 > 1),
    // covering the Conv output range (~150) without clamping. After /10 → y_scale ≈ 0.1.
    float fwd_fq_lo = (quantization_precision == ov::element::u16) ? 0.f : -327680.f;
    float fwd_fq_hi = (quantization_precision == ov::element::u16) ? 655350.f : 327670.f;
    auto fq_pass_il = ov::op::v0::Constant::create(ov::element::f32, {}, {fwd_fq_lo});
    auto fq_pass_ih = ov::op::v0::Constant::create(ov::element::f32, {}, {fwd_fq_hi});
    auto fq_pass_ol = ov::op::v0::Constant::create(ov::element::f32, {}, {fwd_fq_lo});
    auto fq_pass_oh = ov::op::v0::Constant::create(ov::element::f32, {}, {fwd_fq_hi});
    auto fq_pass = std::make_shared<ov::op::v0::FakeQuantize>(dq, fq_pass_il, fq_pass_ih, fq_pass_ol, fq_pass_oh, 65536);

    float branch_fq_lo = (quantization_precision == ov::element::u16) ? 0.f : -32800.f;
    float branch_fq_hi = (quantization_precision == ov::element::u16) ? 65600.f : 32800.f;

    // Helper lambda to create a residual block
    auto create_residual_block = [&](const ov::Output<ov::Node>& input, size_t seed) {
        auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
        // Left branch: MVN -> Conv
        auto mvn = std::make_shared<ov::op::v6::MVN>(input, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

        auto weight = build_dq_subgraph(ov::element::i8, ov::Shape{32, 32, 3, 3}, 0.003f, -128, seed);
        auto conv = std::make_shared<ov::op::v1::Convolution>(mvn,
                                                              weight,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});

        // Bias with DQ (1D bias: [32])
        auto bias = build_dq_subgraph(ov::element::i8, {32}, 0.001f, 0);
        auto conv_biased = add_bias(conv, bias);

        // Insert FQ (65536 levels) on the Conv+bias branch.
        // Backward propagation triggered from forward propagation adjusts this FQ's
        // range constants by /10, then the topological walk strips it (y_scale < 1).
        auto bfq_il = ov::op::v0::Constant::create(ov::element::f32, {}, {branch_fq_lo});
        auto bfq_ih = ov::op::v0::Constant::create(ov::element::f32, {}, {branch_fq_hi});
        auto bfq_ol = ov::op::v0::Constant::create(ov::element::f32, {}, {branch_fq_lo});
        auto bfq_oh = ov::op::v0::Constant::create(ov::element::f32, {}, {branch_fq_hi});
        auto branch_fq = std::make_shared<ov::op::v0::FakeQuantize>(conv_biased, bfq_il, bfq_ih, bfq_ol, bfq_oh, 65536);

        return std::make_shared<ov::op::v1::Add>(branch_fq, input);
    };

    auto add1 = create_residual_block(fq_pass, 2);
    auto add2 = create_residual_block(add1, 3);
    auto add3 = create_residual_block(add2, 4);

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
    auto final_mvn = std::make_shared<ov::op::v6::MVN>(add3, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{final_mvn}, params, "QDQStripping");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
