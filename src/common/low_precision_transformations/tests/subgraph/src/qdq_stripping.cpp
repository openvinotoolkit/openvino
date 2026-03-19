// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/qdq_stripping.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "ov_lpt_models/qdq_stripping.hpp"
#include "transformations/common_optimizations/convert_quantize_dequantize.hpp"

using namespace testing;
using namespace ov::builder::subgraph;

using QDQStrippingTestParams = std::tuple<bool, ov::element::Type>;

class QDQStrippingTest : public TransformationTestsF, public WithParamInterface<QDQStrippingTestParams> {
public:
    static std::string getTestCaseName(const ::testing::TestParamInfo<QDQStrippingTestParams>& info) {
        const auto& [need_weights_adjustment, quantization_precision] = info.param;
        return std::string(need_weights_adjustment ? "WeightsAdjusted" : "WeightsOriginal") + "_" +
               quantization_precision.get_type_name();
    }

    QDQStrippingTest() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }

    void SetUp() override {
        using namespace ov::element;
        TransformationTestsF::SetUp();
        const auto& [need_weights_adjustment, quantization_precision] = GetParam();
        manager.register_pass<ov::pass::ConvertQuantizeDequantize>(TypeVector{i16, u16}, TypeVector{f32}, true);
        manager.register_pass<ov::pass::low_precision::FQStrippingTransformation>(std::set<size_t>{65536},
                                                                                  need_weights_adjustment);
    }
};

TEST_P(QDQStrippingTest, smoke_LPT_SharedDQ) {
    const auto& [need_weights_adjustment, quantization_precision] = GetParam();
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};
    model = QDQStrippingFunction::build_shared_dq_pattern(input_shape, quantization_precision);
    model_ref = QDQStrippingFunction::build_shared_dq_pattern_ref(input_shape, need_weights_adjustment);
}

TEST_P(QDQStrippingTest, NeedScalingMulMatMul) {
    const auto& [need_weights_adjustment, quantization_precision] = GetParam();
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};
    model = QDQStrippingFunction::build_mul_matmul_pattern(input_shape, quantization_precision);
    model_ref = QDQStrippingFunction::build_mul_matmul_pattern_ref(input_shape, need_weights_adjustment);
}

TEST_P(QDQStrippingTest, NeedScalingMatMulWithBias) {
    const auto& [need_weights_adjustment, quantization_precision] = GetParam();
    const auto input_shape = ov::PartialShape{1, 128};
    model = QDQStrippingFunction::build_matmul_with_bias_pattern(input_shape, quantization_precision);
    model_ref = QDQStrippingFunction::build_matmul_with_bias_pattern_ref(input_shape, need_weights_adjustment);
}

TEST_P(QDQStrippingTest, NeedScalingResidualBlock) {
    const auto& [need_weights_adjustment, quantization_precision] = GetParam();
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};
    model = QDQStrippingFunction::build_residual_block_pattern(input_shape, quantization_precision);
    model_ref = QDQStrippingFunction::build_residual_block_pattern_ref(input_shape, need_weights_adjustment);
}

TEST_P(QDQStrippingTest, NeedScalingResidualBlockNoFinalMVN) {
    const auto& [need_weights_adjustment, quantization_precision] = GetParam();
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};
    model = QDQStrippingFunction::build_residual_block_pattern(input_shape,
                                                               quantization_precision,
                                                               /*skip_final_mvn=*/true);
    model_ref = QDQStrippingFunction::build_residual_block_pattern_ref(input_shape, false, true);
}

TEST_P(QDQStrippingTest, NeedScalingForwardBias) {
    const auto& [need_weights_adjustment, quantization_precision] = GetParam();
    const auto input_shape = ov::PartialShape{1, 128};
    model = QDQStrippingFunction::build_forward_bias_pattern(input_shape, quantization_precision);
    model_ref = QDQStrippingFunction::build_forward_bias_pattern_ref(input_shape, need_weights_adjustment);
}

INSTANTIATE_TEST_SUITE_P(smoke_LPT,
                         QDQStrippingTest,
                         Combine(Bool(), Values(ov::element::u16, ov::element::i16)),
                         QDQStrippingTest::getTestCaseName);
