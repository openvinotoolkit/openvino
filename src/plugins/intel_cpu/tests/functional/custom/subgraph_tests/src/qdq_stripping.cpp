// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/fake_quantize.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"

namespace ov {
namespace test {

using ov::test::InputShape;

enum class PatternType { SharedDQ, NeedScalingMulMatMul, NeedScalingResidualBlock };

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

    ov::Output<ov::Node> build_dq(const ov::Output<ov::Node>& input,
                                  const ov::element::Type& quantization_precision) const {
        auto act_zero_point = ov::op::v0::Constant::create(quantization_precision, {}, {zero_point});
        auto act_zp_convert = std::make_shared<ov::op::v0::Convert>(act_zero_point, ov::element::f32);

        auto act_subtract = std::make_shared<ov::op::v1::Subtract>(input, act_zp_convert);
        auto act_scale = ov::op::v0::Constant::create(ov::element::f32, {}, {(i_h - i_l) / (o_h - o_l)});

        return std::make_shared<ov::op::v1::Multiply>(act_subtract, act_scale);
    }

    float i_l;
    float i_h;
    float o_l;
    float o_h;
    int zero_point;
};

class QDQStrippingTest : public testing::WithParamInterface<QDQStrippingParams>,
                         virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QDQStrippingParams>& obj) {
        const auto& [input_shape, input_precision, quantization_precision, pattern_type] = obj.param;
        std::ostringstream result;
        result << "input_shape=" << input_shape << "_input_precision=" << input_precision
               << "_quantization_precision=" << quantization_precision << "_pattern=" << pattern_type;
        return result.str();
    }

protected:
    std::shared_ptr<ov::Model> build_need_scaling_residual_block_pattern(
        const ov::PartialShape& input_shape,
        const ov::element::Type& quantization_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};

        // First convolution
        ov::test::utils::InputGenerateData weights1_gen_data;
        weights1_gen_data.seed = 1;
        auto weight1 = ov::test::utils::make_constant(ov::element::f32, ov::Shape{32, 3, 3, 3}, weights1_gen_data);
        auto conv1 = std::make_shared<ov::op::v1::Convolution>(params[0],
                                                               weight1,
                                                               ov::Strides{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               ov::CoordinateDiff{1, 1},
                                                               ov::Strides{1, 1});

        // QDQ pattern after first convolution
        static const std::unordered_map<ov::element::Type_t, std::pair<QuantizationParams, QuantizationParams>>
            quantization_params{
                {ov::element::Type_t::u16,
                 {{0.f, 10.f, 0.f, 65535.f, 0}, {-6.244578838348389f, 6.347373962402344f, 0.f, 65535.f, 32500}}},
                {ov::element::Type_t::i16,
                 {{-5.000076293945312f, 4.999923706054688f, -32768.f, 32767.f, 0},
                  {-6.296072483062744f, 6.295880317687988f, -32768.f, 32767.f, 0}}},
            };

        const auto& qp = quantization_params.at(quantization_precision).first;
        auto fq = qp.build_fq(conv1);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
        auto dq = qp.build_dq(convert2, quantization_precision);

        // Helper lambda to create a residual block
        auto create_residual_block = [&](const ov::Output<ov::Node>& input, size_t seed) {
            auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
            // Left branch: MVN -> Conv
            auto mvn =
                std::make_shared<ov::op::v6::MVN>(input, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

            ov::test::utils::InputGenerateData weights_gen_data;
            weights_gen_data.seed = seed;
            auto weight = ov::test::utils::make_constant(ov::element::f32, ov::Shape{32, 32, 3, 3}, weights_gen_data);
            auto conv = std::make_shared<ov::op::v1::Convolution>(mvn,
                                                                  weight,
                                                                  ov::Strides{1, 1},
                                                                  ov::CoordinateDiff{1, 1},
                                                                  ov::CoordinateDiff{1, 1},
                                                                  ov::Strides{1, 1});
            return std::make_shared<ov::op::v1::Add>(conv, input);
        };

        auto add1 = create_residual_block(dq, 2);
        auto add2 = create_residual_block(add1, 3);
        auto add3 = create_residual_block(add2, 4);

        auto reduction_axes = ov::op::v0::Constant::create(ov::element::i64, {3}, {1, 2, 3});
        auto final_mvn =
            std::make_shared<ov::op::v6::MVN>(add3, reduction_axes, true, 1e-9f, ov::op::MVNEpsMode::INSIDE_SQRT);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{final_mvn}, params, "QDQStripping");
        return model;
    }

    std::shared_ptr<ov::Model> build_need_scaling_mul_matmul_pattern(const ov::PartialShape& input_shape,
                                                                     const ov::element::Type& quantization_precision) {
        auto param1 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        ov::ParameterVector params{param1, param2};

        // Common constant
        auto common_constant = ov::op::v0::Constant::create(ov::element::f32, {}, {0.1f});

        // param1 * common_constant
        auto mul1 = std::make_shared<ov::op::v1::Multiply>(param1, common_constant);

        // param2 * common_constant
        auto mul2 = std::make_shared<ov::op::v1::Multiply>(param2, common_constant);

        // MatMul
        auto matmul = std::make_shared<ov::op::v0::MatMul>(mul1, mul2, false, true);

        // QDQ pattern
        static const std::unordered_map<ov::element::Type_t, std::pair<QuantizationParams, QuantizationParams>>
            quantization_params{
                {ov::element::Type_t::u16,
                 {{0.f, 10.f, 0.f, 65535.f, 0}, {-6.244578838348389f, 6.347373962402344f, 0.f, 65535.f, 32500}}},
                {ov::element::Type_t::i16,
                 {{-5.000076293945312f, 4.999923706054688f, -32768.f, 32767.f, 0},
                  {-6.296072483062744f, 6.295880317687988f, -32768.f, 32767.f, 0}}},
            };

        const auto& qp = quantization_params.at(quantization_precision).first;
        auto fq = qp.build_fq(matmul);
        auto convert1 = std::make_shared<ov::op::v0::Convert>(fq, quantization_precision);
        auto convert2 = std::make_shared<ov::op::v0::Convert>(convert1, ov::element::f32);
        auto dq = qp.build_dq(convert2, quantization_precision);

        // Softmax
        auto softmax = std::make_shared<ov::op::v8::Softmax>(dq, -1);

        auto model = std::make_shared<ov::Model>(ov::OutputVector{softmax}, params, "QDQStripping");
        return model;
    }

    std::shared_ptr<ov::Model> build_shared_dq_pattern(const ov::PartialShape& input_shape,
                                                       const ov::element::Type& quantization_precision) {
        ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape)};
        // Note: these params are taken from the real cases
        static const std::unordered_map<ov::element::Type_t, std::pair<QuantizationParams, QuantizationParams>>
            quantization_params{
                {ov::element::Type_t::u16,
                 {{0.f, 10.f, 0.f, 65535.f, 0}, {-6.244578838348389f, 6.347373962402344f, 0.f, 65535.f, 32500}}},
                {ov::element::Type_t::i16,
                 {{-5.000076293945312f, 4.999923706054688f, -32768.f, 32767.f, 0},
                  {-6.296072483062744f, 6.295880317687988f, -32768.f, 32767.f, 0}}},
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

    void SetUp() override {
        targetDevice = ov::test::utils::DEVICE_CPU;
        const auto& [input_shape, input_precision, quantization_precision, pattern_type] = GetParam();

        // NeedScalingMulMatMul pattern has 2 parameters, SharedDQ has 1
        if (pattern_type == PatternType::NeedScalingMulMatMul) {
            init_input_shapes({input_shape, input_shape});
        } else {
            init_input_shapes({input_shape});
        }

        inType = outType = input_precision;

        OPENVINO_ASSERT(quantization_precision == ov::element::i16 || quantization_precision == ov::element::u16,
                        "Only i16 and u16 quantization precisions are supported in the test");
        abs_threshold = (pattern_type == PatternType::NeedScalingResidualBlock) ? 3.f : 1e-2f;

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
        default:
            OPENVINO_THROW("Unknown PatternType");
        }
    }

    void validate() override {
        ov::test::SubgraphBaseTest::validate();
        auto runtime_model = compiledModel.get_runtime_model();
        ASSERT_TRUE(runtime_model != nullptr) << "Runtime model should not be null";
        size_t fake_quantize_count = 0;
        for (const auto& op : runtime_model->get_ordered_ops()) {
            if (op->get_type_info() == ov::op::v0::FakeQuantize::get_type_info_static()) {
                fake_quantize_count++;
            }
        }
        // For CPU, we expect FakeQuantize nodes to be optimized away or fused
        const size_t expected_fake_quantize_count = 0;
        ASSERT_EQ(fake_quantize_count, expected_fake_quantize_count) << "Unexpected FakeQuantize node count.";
    }
};

TEST_P(QDQStrippingTest, Inference) {
    run();
}

namespace {
const std::vector<ov::test::InputShape> input_shapes = {{{-1, -1, -1, -1}, {{1, 3, 128, 128}}}};
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};
const std::vector<ov::element::Type> quantization_precisions = {ov::element::u16, ov::element::i16};
const std::vector<PatternType> pattern_types = {PatternType::SharedDQ,
                                                PatternType::NeedScalingMulMatMul,
                                                PatternType::NeedScalingResidualBlock};

INSTANTIATE_TEST_SUITE_P(smoke_QDQStripping,
                         QDQStrippingTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(quantization_precisions),
                                            ::testing::ValuesIn(pattern_types)),
                         QDQStrippingTest::getTestCaseName);
}  // namespace

}  // namespace test
}  // namespace ov
