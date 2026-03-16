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
enum class PatternType { SharedDQ, NeedScalingMulMatMul, NeedScalingResidualBlock, NeedScalingMatMulWithBias, NeedScalingForwardBias };

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
    case PatternType::NeedScalingForwardBias:
        os << "NeedScalingForwardBias";
        break;
    default:
        OPENVINO_THROW("Unknown PatternType");
    }
    return os;
}

using QDQStrippingParams = std::tuple<ov::test::InputShape, ov::element::Type, ov::element::Type, ov::element::Type, PatternType>;

class QDQStrippingTest : public testing::WithParamInterface<QDQStrippingParams>, virtual public ov::test::SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QDQStrippingParams>& obj) {
        using ov::test::operator<<;

        const auto& [input_shape, input_precision, quantization_precision, inference_precision, pattern_type] = obj.param;
        std::ostringstream result;
        result << "input_shape=" << input_shape << "_input_precision=" << input_precision << "_quantization_precision=" << quantization_precision
               << "_inference_precision=" << inference_precision << "_pattern=" << pattern_type;
        return result.str();
    }

    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override {
        const auto& [input_shape, input_precision, quantization_precision, inference_precision, pattern_type] = GetParam();
        // In "NeedScaling" cases we generate values which would cause overflow in f16 if quantization scales are not adjusted by FQStripping
        if (pattern_type == PatternType::NeedScalingMulMatMul) {
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
        } else if (pattern_type == PatternType::NeedScalingMatMulWithBias ||
                   pattern_type == PatternType::NeedScalingForwardBias) {
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
        const auto& [input_shape, input_precision, quantization_precision, inference_precision, pattern_type] = GetParam();

        if (pattern_type == PatternType::NeedScalingMulMatMul) {
            init_input_shapes({input_shape, input_shape});
        } else {
            init_input_shapes({input_shape});
        }

        inType = outType = input_precision;

        // FakeQuantize alters values slightly during computation (scaling, rounding to discrete
        // levels), so when FQ is stripped, minor differences from the reference are expected.
        abs_threshold = 0.05;

        configuration[ov::hint::inference_precision.name()] = inference_precision.get_type_name();

        OPENVINO_ASSERT(quantization_precision == ov::element::i16 || quantization_precision == ov::element::u16,
                        "Only i16 and u16 quantization precisions are supported in the test");
        switch (pattern_type) {
            using ov::builder::subgraph::QDQStrippingFunction;
        case PatternType::SharedDQ:
            function = QDQStrippingFunction::build_shared_dq_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingMulMatMul:
            function = QDQStrippingFunction::build_mul_matmul_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingResidualBlock:
            function = QDQStrippingFunction::build_residual_block_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingMatMulWithBias:
            function = QDQStrippingFunction::build_matmul_with_bias_pattern(input_shape.first, quantization_precision);
            break;
        case PatternType::NeedScalingForwardBias:
            function = QDQStrippingFunction::build_forward_bias_pattern(input_shape.first, quantization_precision);
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

// NeedScalingMulMatMul is an artificial model designed to test f16 overflow handling via weight scaling.
// For f16, weight scaling divides weights by scale_divisor, reducing MatMul output to fit
// within the FQ range — so stripping the FQ is safe (no clamping effect lost).
// For f32, weight scaling is unnecessary (no overflow risk), so MatMul output remains large
// and exceeds the FQ range — stripping the FQ removes clamping and breaks Softmax accuracy.
INSTANTIATE_TEST_SUITE_P(smoke_QDQStripping_f16Only,
                         QDQStrippingTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(quantization_precisions),
                                            ::testing::Values(ov::element::f16),
                                            ::testing::Values(PatternType::NeedScalingMulMatMul)),
                         QDQStrippingTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_QDQStripping_BothPrecisions,
                         QDQStrippingTest,
                         ::testing::Combine(::testing::ValuesIn(input_shapes),
                                            ::testing::ValuesIn(input_precisions),
                                            ::testing::ValuesIn(quantization_precisions),
                                            ::testing::Values(ov::element::f16, ov::element::f32),
                                            ::testing::Values(PatternType::SharedDQ,
                                                              PatternType::NeedScalingResidualBlock,
                                                              PatternType::NeedScalingMatMulWithBias,
                                                              PatternType::NeedScalingForwardBias)),
                         QDQStrippingTest::getTestCaseName);
}  // namespace