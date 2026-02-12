// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/qdq_stripping.hpp"

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "shared_test_classes/base/utils/ranges.hpp"

namespace {
using namespace ov::test;
using ov::builder::subgraph::QDQStrippingFunction;
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
        if (pattern_type == PatternType::NeedScalingMulMatMul) {
            // Input range [0, 1000]: with DQ=127, mul1 = input * 127 → up to 127,000 → overflows f16.
            // After scale adjustment (÷2): mul1 = input * 63.5 → up to 63,500 → fits f16.
            // Small range keeps values moderate after adjustment, improving Softmax accuracy.
            inputs.clear();
            auto itTargetShape = targetInputStaticShapes.begin();
            for (const auto& param : function->get_parameters()) {
                ov::test::utils::InputGenerateData gen_data(0, 1000, 1, 1);
                auto tensor = ov::test::utils::create_and_fill_tensor(param->get_element_type(), *itTargetShape, gen_data);
                inputs.insert({param, tensor});
                itTargetShape++;
            }
        } else if (pattern_type == PatternType::NeedScalingResidualBlock) {
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
        // All patterns use abs_threshold=0.05.
        // NeedScaling* patterns cause f16 overflow without scale adjustment:
        //   - ResidualBlock and MatMulWithBias use MVN (linearly sensitive to FQ rounding).
        //   - MulMatMul uses Softmax over axis=1 (3 elements) with y_scale=2 and input range
        //     [0,1000], so Softmax argmax is stable with tight accuracy.
        // SharedDQ pattern doesn't cause overflow; small FQ ranges (÷10 vs NeedScaling*)
        // keep f16 quantization error well below 0.05.
        abs_threshold = 0.05;

        // Force f16 inference precision to test FQTransformation scales adjustment (preventing overflow in f16 scenarios).
        configuration[ov::hint::inference_precision.name()] = ov::element::f16.get_type_name();

        OPENVINO_ASSERT(quantization_precision == ov::element::i16 || quantization_precision == ov::element::u16,
                        "Only i16 and u16 quantization precisions are supported in the test");
        switch (pattern_type) {
        case PatternType::SharedDQ:
            function = QDQStrippingFunction::build_shared_dq_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingMulMatMul:
            function = QDQStrippingFunction::build_need_scaling_mul_matmul_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingResidualBlock:
            function = QDQStrippingFunction::build_need_scaling_residual_block_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingMatMulWithBias:
            function = QDQStrippingFunction::build_need_scaling_matmul_with_bias_pattern(input_shape.first, quantization_precision);
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
        // All FQs use 65536 levels (in levels_to_strip), so all should be stripped.
        // NeedScalingResidualBlock has 4 additional FQs (1 forward-path + 3 branch),
        // but their ranges are adjusted by propagation so y_scale drops below threshold,
        // and the topological walk strips them without further propagation.
        ASSERT_EQ(quantize_count, 0u) << "Unexpected Quantize node count.";
    }
};

TEST_P(QDQStrippingTest, Inference) {
    run();
}

const std::vector<ov::test::InputShape> input_shapes = {{{-1, -1, -1, -1}, {{1, 3, 128, 128}}}};
const std::vector<ov::element::Type> input_precisions = {ov::element::f32};
const std::vector<ov::element::Type> quantization_precisions = {ov::element::u16, ov::element::i16};
const std::vector<PatternType> pattern_types = {PatternType::SharedDQ,
                                                PatternType::NeedScalingMulMatMul,
                                                PatternType::NeedScalingResidualBlock,
                                                PatternType::NeedScalingMatMulWithBias};

INSTANTIATE_TEST_SUITE_P(smoke_QDQStripping,
                         QDQStrippingTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(quantization_precisions),
                                            ::testing::ValuesIn(pattern_types)),
                         QDQStrippingTest::getTestCaseName);
}  // namespace