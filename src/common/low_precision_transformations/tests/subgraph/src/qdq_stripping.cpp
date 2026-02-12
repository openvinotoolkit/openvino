// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "low_precision/qdq_stripping.hpp"
#include "ov_lpt_models/qdq_stripping.hpp"

using namespace testing;
using namespace ov::builder::subgraph;

// =============================================================================
// SharedDQ: two Conv branches sharing quantized input, FQs with y_scale < 1
// Expected: all FQs stripped, no scale propagation
// =============================================================================

using QDQStrippingSharedDQTest = TransformationTestsF;

TEST_F(QDQStrippingSharedDQTest, CompareWithReference) {
    disable_rt_info_check();
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};

    model = QDQStrippingFunction::getOriginalSharedDQ(input_shape);
    model_ref = QDQStrippingFunction::getReferenceSharedDQ(input_shape);

    manager.register_pass<ov::pass::low_precision::FQStrippingTransformation>(std::set<size_t>{65536});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// =============================================================================
// NeedScalingMulMatMul: two params * shared DQ constant → MatMul → FQ→DQ
// FQ y_scale = 2 → DQ constant scale divided by 2
// =============================================================================

using QDQStrippingNeedScalingMulMatMulTest = TransformationTestsF;

TEST_F(QDQStrippingNeedScalingMulMatMulTest, CompareWithReference) {
    disable_rt_info_check();
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};

    model = QDQStrippingFunction::getOriginalNeedScalingMulMatMul(input_shape);
    model_ref = QDQStrippingFunction::getReferenceNeedScalingMulMatMul(input_shape);

    manager.register_pass<ov::pass::low_precision::FQStrippingTransformation>(std::set<size_t>{65536});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// =============================================================================
// NeedScalingMatMulWithBias: MatMul with weights + bias → FQ→DQ → MVN
// FQ y_scale = 4 → both weight and bias scales divided by 4
// =============================================================================

using QDQStrippingNeedScalingMatMulWithBiasTest = TransformationTestsF;

TEST_F(QDQStrippingNeedScalingMatMulWithBiasTest, CompareWithReference) {
    disable_rt_info_check();
    const auto input_shape = ov::PartialShape{1, 3};

    model = QDQStrippingFunction::getOriginalNeedScalingMatMulWithBias(input_shape);
    model_ref = QDQStrippingFunction::getReferenceNeedScalingMatMulWithBias(input_shape);

    manager.register_pass<ov::pass::low_precision::FQStrippingTransformation>(std::set<size_t>{65536});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

// =============================================================================
// NeedScalingResidualBlock: Conv→bias→FQ→DQ→FQ(fwd)→residual blocks→MVN
// First FQ y_scale=10 → stripped + backward/forward propagation
// Forward-path FQ and branch FQs get ranges adjusted then stripped
// =============================================================================

using QDQStrippingNeedScalingResidualBlockTest = TransformationTestsF;

TEST_F(QDQStrippingNeedScalingResidualBlockTest, CompareWithReference) {
    disable_rt_info_check();
    const auto input_shape = ov::PartialShape{1, 3, 8, 8};

    model = QDQStrippingFunction::getOriginalNeedScalingResidualBlock(input_shape);
    model_ref = QDQStrippingFunction::getReferenceNeedScalingResidualBlock(input_shape);

    manager.register_pass<ov::pass::low_precision::FQStrippingTransformation>(std::set<size_t>{65536});
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}
