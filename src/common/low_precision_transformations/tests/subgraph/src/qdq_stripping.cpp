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

// =============================================================================
// SharedDQ: two Conv branches sharing quantized input, FQs with y_scale < 1
// Expected: all FQs stripped, no scale propagation
// =============================================================================
class QDQStrippingTest : public TransformationTestsF {
public:
    QDQStrippingTest() : TransformationTestsF() {
        disable_rt_info_check();
        manager.register_pass<ov::pass::ConvertQuantizeDequantize>();
        manager.register_pass<ov::pass::low_precision::FQStrippingTransformation>(std::set<size_t>{65536}, true);
        comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    }
};

TEST_F(QDQStrippingTest, smoke_LPT_SharedDQ) {
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};
    model = QDQStrippingFunction::build_shared_dq_pattern(input_shape, ov::element::u16);
    model_ref = QDQStrippingFunction::build_shared_dq_pattern_ref(input_shape);
}

// =============================================================================
// NeedScalingMulMatMul: two params * shared DQ constant → MatMul → FQ→DQ
// FQ y_scale = 2 → DQ constant scale divided by 2
// =============================================================================

TEST_F(QDQStrippingTest, NeedScalingMulMatMul) {
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};
    model = QDQStrippingFunction::build_need_scaling_mul_matmul_pattern(input_shape, ov::element::u16);
    model_ref = QDQStrippingFunction::build_need_scaling_mul_matmul_pattern_ref(input_shape);
}

// =============================================================================
// NeedScalingMatMulWithBias: MatMul with weights + bias → FQ→DQ → MVN
// FQ y_scale = 4 → both weight and bias scales divided by 4
// =============================================================================
TEST_F(QDQStrippingTest, NeedScalingMatMulWithBias) {
    const auto input_shape = ov::PartialShape{1, 128};
    model = QDQStrippingFunction::build_need_scaling_matmul_with_bias_pattern(input_shape, ov::element::u16);
    model_ref = QDQStrippingFunction::build_need_scaling_matmul_with_bias_pattern_ref(input_shape);
}

// =============================================================================
// NeedScalingResidualBlock: Conv→bias→FQ→DQ→FQ(fwd)→residual blocks→MVN
// First FQ y_scale=10 → stripped + backward/forward propagation
// Forward-path FQ and branch FQs get ranges adjusted then stripped
// =============================================================================
TEST_F(QDQStrippingTest, NeedScalingResidualBlock) {
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};
    model = QDQStrippingFunction::build_need_scaling_residual_block_pattern(input_shape, ov::element::u16);
    model_ref = QDQStrippingFunction::build_need_scaling_residual_block_pattern_ref(input_shape);
}
