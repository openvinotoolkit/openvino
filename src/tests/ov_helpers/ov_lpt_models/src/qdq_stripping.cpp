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
ov::Output<ov::Node> QDQStrippingFunction::build_fq(const ov::Output<ov::Node>& input,
                                                    const QuantizationParams& qp,
                                                    size_t levels) {
    auto il = ov::op::v0::Constant::create(ov::element::f32, {}, {qp.i_l});
    auto ih = ov::op::v0::Constant::create(ov::element::f32, {}, {qp.i_h});
    auto ol = ov::op::v0::Constant::create(ov::element::f32, {}, {qp.o_l});
    auto oh = ov::op::v0::Constant::create(ov::element::f32, {}, {qp.o_h});
    return std::make_shared<ov::op::v0::FakeQuantize>(input, il, ih, ol, oh, levels);
}

ov::Output<ov::Node> QDQStrippingFunction::build_dq(const ov::Output<ov::Node>& input,
                                                    const ov::element::Type& quantization_precision,
                                                    const QuantizationParams& qp) {
    auto act_zero_point = ov::op::v0::Constant::create(quantization_precision, {}, {qp.zero_point});
    auto act_zp_convert = std::make_shared<ov::op::v0::Convert>(act_zero_point, ov::element::f32);
    auto act_subtract = std::make_shared<ov::op::v1::Subtract>(input, act_zp_convert);
    float scale_value = (qp.i_h - qp.i_l) / (qp.o_h - qp.o_l);
    auto act_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value});
    return std::make_shared<ov::op::v1::Multiply>(act_subtract, act_scale);
}

ov::Output<ov::Node> QDQStrippingFunction::build_weights_dq(ov::element::Type quantized_type,
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

std::shared_ptr<ov::Model> QDQStrippingFunction::build_shared_dq_pattern(
    const ov::PartialShape& input_shape,
    const ov::element::Type& quantization_precision) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};
    static const std::unordered_map<ov::element::Type_t, std::pair<QuantizationParams, QuantizationParams>>
        quantization_params{
            {ov::element::Type_t::u16,
             {{0.f, 10.f, 0.f, 65535.f, 0}, {-624.4578838348389f, 634.7373962402344f, 0.f, 65535.f, 32500}}},
            {ov::element::Type_t::i16,
             {{-5.0000762939453125f, 4.9999237060546875f, -32768.f, 32767.f, 0},
              {-630.0096435546875f, 629.9904174804688f, -32768.f, 32767.f, 0}}},
        };

    const auto& q_params = quantization_params.at(quantization_precision);
    const auto& qp_1 = q_params.first;
    auto input_fq = build_fq(params[0], qp_1);

    auto input_convert1 = std::make_shared<ov::op::v0::Convert>(input_fq, quantization_precision);
    auto input_convert2 = std::make_shared<ov::op::v0::Convert>(input_convert1, ov::element::f32);

    size_t seed = 1;
    auto create_qdq_branch = [&](float weight_scale_value) {
        auto input_dequantized = build_dq(input_convert2, quantization_precision, qp_1);
        ov::test::utils::InputGenerateData weights_gen_data;
        weights_gen_data.seed = seed;
        auto weight_quantized =
            ov::test::utils::make_constant(ov::element::i8, ov::Shape{32, 3, 3, 3}, weights_gen_data);
        auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight_quantized, ov::element::f32);
        auto weight_scale =
            ov::test::utils::make_constant(ov::element::f32, {}, std::vector<float>{weight_scale_value});
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
        auto fake_quantize = build_fq(conv_biased, qp_2);
        auto act_quantized = std::make_shared<ov::op::v0::Convert>(fake_quantize, quantization_precision);
        auto act_convert = std::make_shared<ov::op::v0::Convert>(act_quantized, ov::element::f32);
        return build_dq(act_convert, quantization_precision, qp_2);
    };

    auto left_branch = create_qdq_branch(1e-4f);
    auto right_branch = create_qdq_branch(1e-5f);
    auto add_branches = std::make_shared<ov::op::v1::Add>(left_branch, right_branch);

    return std::make_shared<ov::Model>(ov::OutputVector{add_branches}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_shared_dq_pattern_ref(const ov::PartialShape& input_shape,
                                                                             bool need_weights_adjustment) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    size_t seed = 1;
    auto create_branch = [&](float weight_scale_value) {
        ov::test::utils::InputGenerateData weights_gen_data;
        weights_gen_data.seed = seed;
        auto weight_quantized =
            ov::test::utils::make_constant(ov::element::i8, ov::Shape{32, 3, 3, 3}, weights_gen_data);
        auto weight_convert = std::make_shared<ov::op::v0::Convert>(weight_quantized, ov::element::f32);
        auto weight_scale =
            ov::test::utils::make_constant(ov::element::f32, {}, std::vector<float>{weight_scale_value});
        auto weight_dequantized = std::make_shared<ov::op::v1::Multiply>(weight_convert, weight_scale);

        auto conv = std::make_shared<ov::op::v1::Convolution>(params[0],
                                                              weight_dequantized,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});

        ov::test::utils::InputGenerateData bias_gen_data(-2.0, 4, 100, seed++);
        auto bias_const = ov::test::utils::make_constant(ov::element::f32, ov::Shape{1, 32, 1, 1}, bias_gen_data);
        auto conv_biased = std::make_shared<ov::op::v1::Add>(conv, bias_const);
        return conv_biased;
    };

    auto left_branch = create_branch(1e-4f);
    auto right_branch = create_branch(1e-5f);
    auto add_branches = std::make_shared<ov::op::v1::Add>(left_branch, right_branch);

    return std::make_shared<ov::Model>(ov::OutputVector{add_branches}, params, "QDQStripping");
}
std::shared_ptr<ov::Model> QDQStrippingFunction::build_mul_matmul_pattern(
    const ov::PartialShape& input_shape,
    const ov::element::Type& quantization_precision) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    ov::ParameterVector params{param1, param2};

    // Preceding Multiply with DQ weights that backward propagation must NOT scale.
    auto preceding_constant = build_weights_dq(ov::element::i8, {}, 1.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_mul1 = std::make_shared<ov::op::v1::Multiply>(param1, preceding_constant);
    auto preceding_mul2 = std::make_shared<ov::op::v1::Multiply>(param2, preceding_constant);

    // Weight DQ pattern: quantized constant -> convert -> subtract (zero point) -> multiply (scale)
    // DQ = (127 - 0) * 1.0 = 127. Large constant ensures overflow without scale adjustment:
    // input (up to 1000) * 127 = 127,000 >> f16 max (65504)
    auto common_constant = build_weights_dq(ov::element::i8, {}, 1.0f, 0, std::nullopt, std::vector<int>{127});

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(preceding_mul1, common_constant);
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(preceding_mul2, common_constant);

    // Softmax on mul1 detects overflow: if mul1 contains inf, Softmax produces NaN.
    auto softmax_mul1 = std::make_shared<ov::op::v8::Softmax>(mul1, 1);
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
    auto fq = build_fq(matmul, qp);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, quantization_precision, qp);

    auto softmax = std::make_shared<ov::op::v8::Softmax>(dq, 1);
    return std::make_shared<ov::Model>(ov::OutputVector{softmax_mul1, softmax}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_mul_matmul_pattern_ref(const ov::PartialShape& input_shape,
                                                                              bool need_weights_adjustment) {
    auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
    ov::ParameterVector params{param1, param2};

    // Preceding Multiply: NOT scaled (backward propagation must stop at the main Multiply).
    auto preceding_constant = build_weights_dq(ov::element::i8, {}, 1.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_mul1 = std::make_shared<ov::op::v1::Multiply>(param1, preceding_constant);
    auto preceding_mul2 = std::make_shared<ov::op::v1::Multiply>(param2, preceding_constant);

    // Original scale=1.0. With weights adjustment: divided by scale_divisor=20 → 0.05
    const float common_scale = need_weights_adjustment ? 1.0f / 20.0f : 1.0f;
    auto common_constant = build_weights_dq(ov::element::i8, {}, common_scale, 0, std::nullopt, std::vector<int>{127});

    auto mul1 = std::make_shared<ov::op::v1::Multiply>(preceding_mul1, common_constant);
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(preceding_mul2, common_constant);

    auto softmax_mul1 = std::make_shared<ov::op::v8::Softmax>(mul1, 1);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(mul1, mul2, false, true);
    auto softmax = std::make_shared<ov::op::v8::Softmax>(matmul, 1);

    return std::make_shared<ov::Model>(ov::OutputVector{softmax_mul1, softmax}, params, "QDQStripping");
}
std::shared_ptr<ov::Model> QDQStrippingFunction::build_matmul_with_bias_pattern(
    const ov::PartialShape& input_shape,
    const ov::element::Type& quantization_precision) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // Preceding MatMul+bias that backward propagation must NOT scale.
    auto preceding_weight =
        build_weights_dq(ov::element::i8, ov::Shape{128, 128}, 1.0f / 128.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_matmul = std::make_shared<ov::op::v0::MatMul>(params[0], preceding_weight, false, false);
    auto preceding_bias = build_weights_dq(ov::element::i8, {128}, 1.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_matmul_biased = std::make_shared<ov::op::v1::Add>(preceding_matmul, preceding_bias);

    // Weight DQ for MatMul: input last dim = 128 (from input shape [1,3,128,128]), output = 32
    // Zero-point -128 shifts i8[-128,127] to [0,255]. Scale=0.02 → weights in [0, 5.1].
    // All-positive weights + all-positive inputs [0,100] → MatMul output always positive.
    // This avoids u16 FQ clamping (input_low=0) which would cause large f16 rounding errors.
    // With 128-element dot product: avg output ≈ 50 * 2.55 * 128 ≈ 16320.
    auto weight = build_weights_dq(ov::element::i8, ov::Shape{128, 32}, 0.02f, -128, 1);

    auto matmul = std::make_shared<ov::op::v0::MatMul>(preceding_matmul_biased, weight, false, false);

    // Bias DQ: [32] with zero_point=-128 so DQ values are always non-negative.
    // DQ = (i8_val + 128) * 200 → [0, 51000], avg ~25500.
    // Bias is significant relative to MatMul output (~16320) so MVN detects unscaled bias.
    auto bias = build_weights_dq(ov::element::i8, {32}, 200.0f, -128, 2);
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
    auto fq = build_fq(matmul_biased, qp);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, quantization_precision, qp);

    // MVN is scale-invariant (normalizes by mean/variance), triggering scale adjustment.
    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mvn = std::make_shared<ov::op::v6::MVN>(dq, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);
    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_matmul_with_bias_pattern_ref(const ov::PartialShape& input_shape,
                                                                                    bool need_weights_adjustment) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // Original weight_scale=0.02. With weights adjustment: divided by scale_divisor=40 → 0.0005
    const float scale_divisor = need_weights_adjustment ? 40.f : 1.f;

    // Preceding MatMul+bias: NOT scaled (backward propagation must stop at the main MatMul+bias).
    auto preceding_weight =
        build_weights_dq(ov::element::i8, ov::Shape{128, 128}, 1.0f / 128.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_matmul = std::make_shared<ov::op::v0::MatMul>(params[0], preceding_weight, false, false);
    auto preceding_bias = build_weights_dq(ov::element::i8, {128}, 1.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_matmul_biased = std::make_shared<ov::op::v1::Add>(preceding_matmul, preceding_bias);

    auto weight = build_weights_dq(ov::element::i8, ov::Shape{128, 32}, 0.02f / scale_divisor, -128, 1);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(preceding_matmul_biased, weight, false, false);

    // Original bias_scale=200. With weights adjustment: divided by 40 → 5
    auto bias = build_weights_dq(ov::element::i8, {32}, 200.0f / scale_divisor, -128, 2);
    auto matmul_biased = std::make_shared<ov::op::v1::Add>(matmul, bias);

    // FQ stripped, CQDQ removed Convert+DQ → Add output goes directly to MVN
    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(matmul_biased, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "QDQStripping");
}
std::shared_ptr<ov::Model> QDQStrippingFunction::build_residual_block_pattern(
    const ov::PartialShape& input_shape,
    const ov::element::Type& quantization_precision,
    bool skip_final_mvn) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // Preceding Conv+bias that backward propagation must NOT scale.
    // It sits before the FQ being stripped, separated by Conv1+bias. If backward
    // propagation incorrectly continues past Conv1+bias, it would reach this Conv
    // and erroneously divide its weights by scale_divisor.
    auto preceding_weight =
        build_weights_dq(ov::element::i8, ov::Shape{3, 3, 3, 3}, 1.0f / 27.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_conv = std::make_shared<ov::op::v1::Convolution>(params[0],
                                                                    preceding_weight,
                                                                    ov::Strides{1, 1},
                                                                    ov::CoordinateDiff{1, 1},
                                                                    ov::CoordinateDiff{1, 1},
                                                                    ov::Strides{1, 1});
    auto preceding_bias = build_weights_dq(ov::element::i8, {3}, 1.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_conv_biased = add_bias(preceding_conv, preceding_bias);

    // First convolution with weight DQ
    // zp=-128 shifts i8 range to [0, 255] → all-positive DQ weights,
    // avoiding u16 FQ clamping at input_low=0 which causes large errors with mixed-sign weights.
    // scale=0.003 produces DQ values up to ~0.765, large enough to cause f16 overflow
    // after y_scale=10 FQ, validating the scale adjustment logic.
    auto weight1 = build_weights_dq(ov::element::i8, ov::Shape{32, 3, 3, 3}, 0.003f, -128, 1);
    auto conv1 = std::make_shared<ov::op::v1::Convolution>(preceding_conv_biased,
                                                           weight1,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::Strides{1, 1});

    auto bias1 = build_weights_dq(ov::element::i8, {32}, 0.001f, 0);
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
    auto fq = build_fq(conv1_biased, qp);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, quantization_precision, qp);

    // Forward-path FQ→DQ chain: uses same qp so y_scale=10
    // After stripping, forward propagation adjusts downstream FQ ranges by scale_divisor.
    auto fq_pass = build_fq(dq, qp);
    auto fq_pass_convert1 = std::make_shared<ov::op::v0::Convert>(fq_pass, quantization_precision);
    auto fq_pass_convert2 = std::make_shared<ov::op::v0::Convert>(fq_pass_convert1, ov::element::f32);
    auto fq_pass_dq = build_dq(fq_pass_convert2, quantization_precision, qp);

    // Branch FQ range is [0, 65600] / [-32800, 32800] (y_scale ≈ 1.001 > 1),
    // covering the Conv output range (~150) without clamping. After /10 → y_scale ≈ 0.1.
    float branch_fq_lo = (quantization_precision == ov::element::u16) ? 0.f : -32800.f;
    float branch_fq_hi = (quantization_precision == ov::element::u16) ? 65600.f : 32800.f;

    // Helper lambda to create a residual block
    auto create_residual_block = [&](const ov::Output<ov::Node>& input, size_t seed) {
        auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
        // Left branch: MVN -> Conv
        auto mvn =
            std::make_shared<ov::op::v6::MVN>(input, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);

        auto weight = build_weights_dq(ov::element::i8, ov::Shape{32, 32, 3, 3}, 0.003f, -128, seed);
        auto conv = std::make_shared<ov::op::v1::Convolution>(mvn,
                                                              weight,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});

        // Bias with DQ (1D bias: [32])
        auto bias = build_weights_dq(ov::element::i8, {32}, 0.001f, 0);
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

    auto add1 = create_residual_block(fq_pass_dq, 2);
    auto add2 = create_residual_block(add1, 3);
    auto add3 = create_residual_block(add2, 4);

    if (skip_final_mvn) {
        return std::make_shared<ov::Model>(ov::OutputVector{add3}, params, "QDQStripping");
    }

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
    auto final_mvn =
        std::make_shared<ov::op::v6::MVN>(add3, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{final_mvn}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_residual_block_pattern_ref(const ov::PartialShape& input_shape,
                                                                                  bool need_weights_adjustment,
                                                                                  bool skip_final_mvn) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // y_scale(10) * ratio(10) = 100. When need_weights_adjustment=false, no division.
    const float scale_divisor = need_weights_adjustment ? 100.f : 1.f;

    // Preceding Conv+bias: NOT scaled (backward propagation must stop at Conv1+bias).
    auto preceding_weight =
        build_weights_dq(ov::element::i8, ov::Shape{3, 3, 3, 3}, 1.0f / 27.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_conv = std::make_shared<ov::op::v1::Convolution>(params[0],
                                                                    preceding_weight,
                                                                    ov::Strides{1, 1},
                                                                    ov::CoordinateDiff{1, 1},
                                                                    ov::CoordinateDiff{1, 1},
                                                                    ov::Strides{1, 1});
    auto preceding_bias = build_weights_dq(ov::element::i8, {3}, 1.0f, 0, std::nullopt, std::vector<int>{1});
    auto preceding_conv_biased = add_bias(preceding_conv, preceding_bias);

    // Conv1 weight scale: 0.003/100 = 3e-05
    auto weight1 = build_weights_dq(ov::element::i8, ov::Shape{32, 3, 3, 3}, 0.003f / scale_divisor, -128, 1);
    auto conv1 = std::make_shared<ov::op::v1::Convolution>(preceding_conv_biased,
                                                           weight1,
                                                           ov::Strides{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::CoordinateDiff{1, 1},
                                                           ov::Strides{1, 1});

    // Bias1 scale: 0.001/100 = 1e-05
    auto bias1 = build_weights_dq(ov::element::i8, {32}, 0.001f / scale_divisor, 0);
    auto conv1_biased = add_bias(conv1, bias1);

    // Residual blocks: all branch FQs stripped, all weight/bias scales divided by 100
    auto create_residual_block = [&](const ov::Output<ov::Node>& input, size_t seed) {
        auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
        auto mvn =
            std::make_shared<ov::op::v6::MVN>(input, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);

        // Weight scale: 0.003/100 = 3e-05
        auto weight = build_weights_dq(ov::element::i8, ov::Shape{32, 32, 3, 3}, 0.003f / scale_divisor, -128, seed);
        auto conv = std::make_shared<ov::op::v1::Convolution>(mvn,
                                                              weight,
                                                              ov::Strides{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::CoordinateDiff{1, 1},
                                                              ov::Strides{1, 1});

        // Bias scale: 0.001/100 = 1e-05
        auto bias = build_weights_dq(ov::element::i8, {32}, 0.001f / scale_divisor, 0);
        auto conv_biased = add_bias(conv, bias);

        // Branch FQ stripped
        return std::make_shared<ov::op::v1::Add>(conv_biased, input);
    };

    auto add1 = create_residual_block(conv1_biased, 2);
    auto add2 = create_residual_block(add1, 3);
    auto add3 = create_residual_block(add2, 4);

    if (skip_final_mvn) {
        return std::make_shared<ov::Model>(ov::OutputVector{add3}, params, "QDQStripping");
    }

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
    auto final_mvn =
        std::make_shared<ov::op::v6::MVN>(add3, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{final_mvn}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_forward_bias_pattern(
    const ov::PartialShape& input_shape,
    const ov::element::Type& quantization_precision) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    // MatMul1 with DQ weights (backward propagation will adjust these)
    auto weight1 = build_weights_dq(ov::element::i8, ov::Shape{128, 32}, 0.02f, -128, 1);
    auto matmul1 = std::make_shared<ov::op::v0::MatMul>(params[0], weight1, false, false);

    auto bias1 = build_weights_dq(ov::element::i8, {32}, 200.0f, -128, 2);
    auto matmul1_biased = std::make_shared<ov::op::v1::Add>(matmul1, bias1);

    // FQ with y_scale = 4 (same as MatMulWithBias pattern)
    static const std::unordered_map<ov::element::Type_t, QuantizationParams> quantization_params{
        {ov::element::Type_t::u16, {0.f, 262140.f, 0.f, 65535.f, 0}},
        {ov::element::Type_t::i16, {-131072.f, 131068.f, -32768.f, 32767.f, 0}},
    };

    const auto& qp = quantization_params.at(quantization_precision);
    auto fq = build_fq(matmul1_biased, qp);
    auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
    auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
    auto dq = build_dq(convert2, quantization_precision, qp);

    // MatMul2 with DQ weights (forward propagation must NOT adjust these weights,
    // but MUST adjust bias2).
    // w2_scale=0.001 keeps MatMul2 output moderate after scale adjustment:
    //   adjusted signal (~1500) × 0.128 × 32 ≈ 6000 → fits f16.
    // bias2_scale=100 is significant relative to MatMul2 output, so MVN detects
    //   unscaled bias2 if forward propagation doesn't adjust it.
    auto weight2 = build_weights_dq(ov::element::i8, ov::Shape{32, 32}, 0.001f, -128, 3);
    auto matmul2 = std::make_shared<ov::op::v0::MatMul>(dq, weight2, false, false);

    auto bias2 = build_weights_dq(ov::element::i8, {32}, 100.0f, -128, 4);
    auto matmul2_biased = std::make_shared<ov::op::v1::Add>(matmul2, bias2);

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(matmul2_biased, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);

    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "QDQStripping");
}

std::shared_ptr<ov::Model> QDQStrippingFunction::build_forward_bias_pattern_ref(const ov::PartialShape& input_shape,
                                                                                bool need_weights_adjustment) {
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

    const float scale_divisor = need_weights_adjustment ? 40.f : 1.f;

    // MatMul1: weight and bias adjusted by backward propagation
    auto weight1 = build_weights_dq(ov::element::i8, ov::Shape{128, 32}, 0.02f / scale_divisor, -128, 1);
    auto matmul1 = std::make_shared<ov::op::v0::MatMul>(params[0], weight1, false, false);

    auto bias1 = build_weights_dq(ov::element::i8, {32}, 200.0f / scale_divisor, -128, 2);
    auto matmul1_biased = std::make_shared<ov::op::v1::Add>(matmul1, bias1);

    // MatMul2: weight NOT adjusted, bias2 adjusted by forward propagation
    auto weight2 = build_weights_dq(ov::element::i8, ov::Shape{32, 32}, 0.001f, -128, 3);
    auto matmul2 = std::make_shared<ov::op::v0::MatMul>(matmul1_biased, weight2, false, false);

    auto bias2 = build_weights_dq(ov::element::i8, {32}, 100.0f / scale_divisor, -128, 4);
    auto matmul2_biased = std::make_shared<ov::op::v1::Add>(matmul2, bias2);

    auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto mvn =
        std::make_shared<ov::op::v6::MVN>(matmul2_biased, reduction_axes, true, 1e-3f, ov::op::MVNEpsMode::INSIDE_SQRT);
    return std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "QDQStripping");
}
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
