// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gtest/gtest.h>
#include "nodes/eltwise.h"

using namespace ov::intel_cpu;

class EltwisePrecisionHelperTest : public testing::Test {};

namespace EltwisePrecisionHelper {
static const std::vector<ov::element::Type> exec_precisions_priority =
        {ov::element::u8, ov::element::bf16, ov::element::i32, ov::element::f32};
}  // namespace EltwisePrecisionHelper

TEST(EltwisePrecisionHelperTest, get_precision_mixed) {
    ov::element::Type src_prc[MAX_ELTWISE_INPUTS];
    const size_t inputs_size = 4ull;
    for (size_t i = 0; i < inputs_size; ++i) {
        src_prc[i] = ov::element::i32;
    }

    std::vector<ov::intel_cpu::EltwiseData> eltwise_data = {
        {Algorithm::EltwiseMultiply},
        {Algorithm::EltwiseMulAdd}
    };

    const auto precision =
        ov::intel_cpu::eltwise_precision_helper::get_precision(inputs_size, src_prc, eltwise_data, EltwisePrecisionHelper::exec_precisions_priority);
    ASSERT_EQ(ov::element::i32, precision);
}

TEST(EltwisePrecisionHelperTest, get_precision_single) {
    ov::element::Type src_prc[MAX_ELTWISE_INPUTS];
    const size_t inputs_size = 4ull;
    for (size_t i = 0; i < inputs_size; ++i) {
        src_prc[i] = ov::element::i32;
    }

    std::vector<ov::intel_cpu::EltwiseData> eltwise_data = {
        {Algorithm::EltwiseMultiply},
        {Algorithm::EltwiseMod}
    };

    const auto precision =
        ov::intel_cpu::eltwise_precision_helper::get_precision(inputs_size, src_prc, eltwise_data, EltwisePrecisionHelper::exec_precisions_priority);
    ASSERT_EQ(ov::element::f32, precision);
}
