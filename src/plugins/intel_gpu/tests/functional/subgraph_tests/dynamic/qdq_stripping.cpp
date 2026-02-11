// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace {
using namespace ov::test;
using ov::test::InputShape;

enum class PatternType { SharedDQ, NeedScalingMulMatMul, NeedScalingResidualBlock, NeedScalingMatMulWithBias };

inline std::ostream& operator<<(std::ostream& os, PatternType pattern_type) {
    switch (pattern_type) {
    case PatternType::SharedDQ:
        os << "SharedDQ";
        break;
    case PatternType::NeedScalingMulMatMul:
        os << "NeedScalingMulMatMul";
        break;
    case PatternType::NeedScalingResidualBlock:
        os << "NeedScalingResidualBlock";
        break;
    case PatternType::NeedScalingMatMulWithBias:
        os << "NeedScalingMatMulWithBias";
        break;
    default:
        OPENVINO_THROW("Unknown PatternType");
    }
    return os;
}

using QDQStrippingParams = std::tuple<ov::test::InputShape, ov::element::Type, ov::element::Type, PatternType>;

class QuantizationParams {
public:
    ov::Output<ov::Node> build_fq(const ov::Output<ov::Node>& input) const {
        auto input_low = ov::op::v0::Constant::create(ov::element::f32, {}, {i_l});
        auto input_high = ov::op::v0::Constant::create(ov::element::f32, {}, {i_h});
        auto output_low = ov::op::v0::Constant::create(ov::element::f32, {}, {o_l});
        auto output_high = ov::op::v0::Constant::create(ov::element::f32, {}, {o_h});
        return std::make_shared<ov::op::v0::FakeQuantize>(input, input_low, input_high, output_low, output_high, 65536);
    }

    ov::Output<ov::Node> build_dq(const ov::Output<ov::Node>& input, const ov::element::Type& quantization_precision) const {
        auto act_zero_point = ov::op::v0::Constant::create(quantization_precision, {}, {zero_point});
        auto act_zp_convert = std::make_shared<ov::op::v0::Convert>(act_zero_point, ov::element::f32);

        auto act_subtract = std::make_shared<ov::op::v1::Subtract>(input, act_zp_convert);
        float scale_value = (i_h - i_l) / (o_h - o_l);
        auto act_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value});

        return std::make_shared<ov::op::v1::Multiply>(act_subtract, act_scale);
    }

    float i_l;
    float i_h;
    float o_l;
    float o_h;
    int zero_point;
};

class QDQStrippingTest : public testing::WithParamInterface<QDQStrippingParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QDQStrippingParams>& obj) {
        const auto& [input_shape, input_precision, quantization_precision, pattern_type] = obj.param;
        std::ostringstream result;
        result << "IS=(" << ov::test::utils::partialShape2str({input_shape.first}) << ")_"
               << "TS=";
        for (const auto& ts : input_shape.second) {
            result << "(" << ov::test::utils::vec2str(ts) << ")_";
        }
        result << "Precision=" << input_precision << "_QuantPrecision=" << quantization_precision << "_Pattern=" << pattern_type;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const auto& [input_shape, input_precision, quantization_precision, pattern_type] = GetParam();
        // In "NeedScaling" cases we generate values which would cause overflow in f16 if quantization scales are not adjusted by FQStripping
        if (pattern_type == PatternType::NeedScalingMulMatMul || pattern_type == PatternType::NeedScalingResidualBlock) {
            inputs.clear();
            auto itTargetShape = targetInputStaticShapes.begin();
            for (const auto& param : function->get_parameters()) {
                auto gen_data = ov::test::utils::rangeByType.get_range(quantization_precision);
                gen_data.range = 65000;
                auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), *itTargetShape, gen_data);
                inputs.insert({param, tensor});
                itTargetShape++;
            }
        } else if (pattern_type == PatternType::NeedScalingMatMulWithBias) {
            // Input range [0, 100]: with weight_scale=0.02, weight_zp=-128, weights in [0,5.1],
            // 128-element dot product → MatMul output ~16320 (all positive).
            // Plus bias [0, 51000] → total up to ~67320 → overflows f16 without scale adj.
            // With scale adjustment (÷4): total ~16830 → fits f16.
            // Positive inputs + positive weights → all-positive signal, no u16 FQ clamping.
            inputs.clear();
            auto itTargetShape = targetInputStaticShapes.begin();
            for (const auto& param : function->get_parameters()) {
                ov::test::utils::InputGenerateData gen_data(0, 100, 1, 1);
                auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), *itTargetShape, gen_data);
                inputs.insert({param, tensor});
                itTargetShape++;
            }
        } else {
            SubgraphBaseTest::generate_inputs(targetInputStaticShapes);
        }
    }

protected:
    ov::Output<ov::Node> build_dq_subgraph(ov::element::Type quantized_type,
                                           const ov::Shape& shape,
                                           float scale_value,
                                           int zero_point = 0,
                                           std::optional<size_t> seed = std::nullopt,
                                           std::optional<std::vector<int>> constant_values = std::nullopt) {
        std::shared_ptr<ov::Node> quantized_const;

        if (seed.has_value()) {
            auto gen_data = ov::test::utils::rangeByType.get_range(quantized_type);
            gen_data.seed = seed.value();
            quantized_const = ov::test::utils::make_constant(quantized_type, shape, gen_data);
        } else if (constant_values.has_value()) {
            quantized_const = ov::test::utils::make_constant(quantized_type, shape, constant_values.value());
        } else {
            // Default: single value 10
            quantized_const = ov::op::v0::Constant::create(quantized_type, shape, {10});
        }

        auto convert = std::make_shared<ov::op::v0::Convert>(quantized_const, ov::element::f32);

        auto zp_quantized = ov::op::v0::Constant::create(quantized_type, {}, {zero_point});
        auto zp_convert = std::make_shared<ov::op::v0::Convert>(zp_quantized, ov::element::f32);
        std::shared_ptr<ov::Node> result = std::make_shared<ov::op::v1::Subtract>(convert, zp_convert);

        auto scale = ov::op::v0::Constant::create(ov::element::f32, {}, {scale_value});
        result = std::make_shared<ov::op::v1::Multiply>(result, scale);

        return result;
    }

    // Helper to build realistic bias pattern: reshapes 1D bias to [1, C, 1, 1, ...] based on conv rank
    ov::Output<ov::Node> add_bias(const ov::Output<ov::Node>& conv, const ov::Output<ov::Node>& bias) {
        const auto conv_shape = std::make_shared<ov::op::v3::ShapeOf>(conv);
        const auto conv_rank = std::make_shared<ov::op::v3::ShapeOf>(conv_shape);

        // Prepare tail shape (rank = conv.rank - 2): [1, 1, 1, 1, ... ]
        const auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto two_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        const auto tail_shape_rank = std::make_shared<ov::op::v1::Subtract>(conv_rank, two_const);
        const auto tail_shape = std::make_shared<ov::op::v3::Broadcast>(one_const, tail_shape_rank);

        // Construct new bias shape: [1, C, 1, 1, ... ]
        const auto C_dim = std::make_shared<ov::op::v3::ShapeOf>(bias);
        const auto new_shape = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one_const, C_dim, tail_shape}, 0);
        const auto reshaped_bias = std::make_shared<ov::op::v1::Reshape>(bias, new_shape, false);

        return std::make_shared<ov::op::v1::Add>(conv, reshaped_bias);
    }

    std::shared_ptr<ov::Model> build_shared_dq_pattern(const ov::PartialShape& input_shape, const ov::element::Type& quantization_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};
        static const std::unordered_map<ov::element::Type_t, std::pair<QuantizationParams, QuantizationParams>> quantization_params{
            {ov::element::Type_t::u16, {{0.f, 10.f, 0.f, 65535.f, 0}, {-6244.578838348389f, 6347.373962402344f, 0.f, 65535.f, 32500}}},
            {ov::element::Type_t::i16, {{-5.f, 5.f, -32768.f, 32767.f, 0}, {-6296.072483062744f, 6295.880317687988f, -32768.f, 32767.f, 0}}},
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

        auto left_branch = create_qdq_branch(1e-3f);
        auto right_branch = create_qdq_branch(1e-4f);
        auto add_branches = std::make_shared<ov::op::v1::Add>(left_branch, right_branch);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{add_branches}, params, "QDQStripping");
        return model;
    }

    std::shared_ptr<ov::Model> build_need_scaling_mul_matmul_pattern(const ov::PartialShape& input_shape, const ov::element::Type& quantization_precision) {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        ov::ParameterVector params{param1, param2};

        // Weight DQ pattern: quantized constant -> convert -> subtract (zero point) -> multiply (scale)
        auto common_constant = build_dq_subgraph(ov::element::i8, {}, 0.1f, 10, std::nullopt, std::vector<int>{100});

        // param1 * common_constant
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(param1, common_constant);

        // param2 * common_constant
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(param2, common_constant);

        // Add Softmax directly on mul1 as first output to detect overflow before MatMul clamps it
        // If mul1 contains inf (due to f16 overflow), Softmax will produce NaN
        auto softmax_mul1 = std::make_shared<ov::op::v8::Softmax>(mul1, -1);

        // MatMul
        auto matmul = std::make_shared<ov::op::v0::MatMul>(mul1, mul2, false, true);

        // y_scale = (input_high - input_low) / (levels - 1) ≈ 655350 / 65535 ≈ 10
        // After DQ: quantized_value * 10 can reach ~655350, far beyond f16 max (65504)
        // Without scale adjustment: Softmax receives inf -> exp(inf) = inf -> inf/inf = NaN
        // With scale adjustment: weights are divided by ~10, keeping values in f16 range
        static const std::unordered_map<ov::element::Type_t, QuantizationParams> quantization_params{
            {ov::element::Type_t::u16, {0.f, 655350.f, 0.f, 65535.f, 0}},
            {ov::element::Type_t::i16, {-327675.f, 327675.f, -32768.f, 32767.f, 0}},
        };

        const auto& qp = quantization_params.at(quantization_precision);
        auto fq = qp.build_fq(matmul);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
        auto dq = qp.build_dq(convert2, quantization_precision);

        auto softmax = std::make_shared<ov::op::v8::Softmax>(dq, -1);
        auto model = std::make_shared<ov::Model>(ov::OutputVector{softmax_mul1, softmax}, params, "QDQStripping");
        return model;
    }

    std::shared_ptr<ov::Model> build_need_scaling_matmul_with_bias_pattern(const ov::PartialShape& input_shape, const ov::element::Type& quantization_precision) {
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
            {ov::element::Type_t::i16, {-131070.f, 131070.f, -32768.f, 32767.f, 0}},
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

        auto model = std::make_shared<ov::Model>(ov::OutputVector{mvn}, params, "QDQStripping");
        return model;
    }

    std::shared_ptr<ov::Model> build_need_scaling_residual_block_pattern(const ov::PartialShape& input_shape, const ov::element::Type& quantization_precision) {
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
            {ov::element::Type_t::i16, {-327675.f, 327675.f, -32768.f, 32767.f, 0}},
        };

        const auto& qp = quantization_params.at(quantization_precision);
        auto fq = qp.build_fq(conv1_biased);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
        auto dq = qp.build_dq(convert2, quantization_precision);

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

            return std::make_shared<ov::op::v1::Add>(conv_biased, input);
        };

        auto add1 = create_residual_block(dq, 2);
        auto add2 = create_residual_block(add1, 3);
        auto add3 = create_residual_block(add2, 4);

        auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
        auto final_mvn = std::make_shared<ov::op::v6::MVN>(add3, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{final_mvn}, params, "QDQStripping");
        return model;
    }

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_GPU;
        const auto& [input_shape, input_precision, quantization_precision, pattern_type] = GetParam();

        // NeedScalingMulMatMul pattern has 2 parameters, others have 1
        if (pattern_type == PatternType::NeedScalingMulMatMul) {
            init_input_shapes({input_shape, input_shape});
        } else {
            init_input_shapes({input_shape});
        }

        inType = outType = input_precision;

        // abs_threshold rationale:
        // - NeedScalingMulMatMul uses Softmax (exponentially sensitive to FQ rounding),
        //   so abs_threshold=1 validates "no NaN" rather than exact f32 match.
        // - NeedScalingResidualBlock uses MVN (linearly sensitive) with all-positive
        //   weights (zp=-128) and y_scale=10 (FQ step=10), giving moderate FQ error.
        // - NeedScalingMatMulWithBias uses MVN with y_scale=4 (FQ step=4),
        //   so FQ error is small relative to signal (~17000), allowing tight accuracy.
        // - SharedDQ uses Softmax, so threshold matches MulMatMul.
        if (pattern_type == PatternType::NeedScalingMatMulWithBias) {
            abs_threshold = 0.05;
        } else if (pattern_type == PatternType::NeedScalingResidualBlock) {
            abs_threshold = 0.05;
        } else {
            abs_threshold = 1;
        }

        // Force f16 inference precision to test FQTransformation scales adjustment (preventing overflow in f16 scenarios).
        configuration[ov::hint::inference_precision.name()] = ov::element::f16.get_type_name();

        OPENVINO_ASSERT(quantization_precision == ov::element::i16 || quantization_precision == ov::element::u16,
                        "Only i16 and u16 quantization precisions are supported in the test");
        switch (pattern_type) {
        case PatternType::SharedDQ:
            function = build_shared_dq_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingMulMatMul:
            function = build_need_scaling_mul_matmul_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingResidualBlock:
            function = build_need_scaling_residual_block_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingMatMulWithBias:
            function = build_need_scaling_matmul_with_bias_pattern(input_shape.first, quantization_precision);
            break;
        default:
            OPENVINO_THROW("Unknown PatternType");
        }
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_TRUE(runtime_model != nullptr) << "Runtime model should not be null";
        size_t quantize_count = 0;
        for (const auto& op : runtime_model->get_ordered_ops()) {
            auto layer_type = op->get_rt_info().at(ov::exec_model_info::LAYER_TYPE).as<std::string>();
            if (layer_type == std::string("Quantize")) {
                quantize_count++;
            }
        }
        const size_t expected_quantize_count = 0;
        ASSERT_EQ(quantize_count, expected_quantize_count) << "Unexpected Quantize node count.";
    }
};

TEST_P(QDQStrippingTest, Inference) {
    run();
}

const std::vector<ov::test::InputShape> input_shapes = {{{-1, -1, -1, -1}, {{1, 3, 128, 128}}}};
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};
const std::vector<ov::element::Type> quantization_precisions = {ov::element::u16, ov::element::i16};
const std::vector<PatternType> pattern_types = {PatternType::SharedDQ, PatternType::NeedScalingMulMatMul, PatternType::NeedScalingResidualBlock, PatternType::NeedScalingMatMulWithBias};

INSTANTIATE_TEST_SUITE_P(smoke_QDQStripping,
                         QDQStrippingTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(quantization_precisions),
                                            ::testing::ValuesIn(pattern_types)),
                         QDQStrippingTest::getTestCaseName);
}  // namespace