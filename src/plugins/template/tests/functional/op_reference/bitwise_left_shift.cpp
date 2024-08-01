// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_left_shift.hpp"

#include <gtest/gtest.h>

#include "bitwise.hpp"

using namespace ov;

namespace reference_tests {
namespace BitwiseOpsRefTestDefinitions {
namespace {

std::vector<RefBitwiseParams> generateBitwiseParams() {
    std::vector<RefBitwiseParams> bitwiseParams{
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{10}, element::i32, std::vector<int32_t>{0, -0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1}, element::i32, std::vector<int32_t>{1}}})
            .expected({{10}, element::i32, std::vector<int32_t>{0, 0, 2, -2, 6, -6, 4, -4, 128, -128}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{10}, element::i8, std::vector<int8_t>{0, -0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1}, element::i8, std::vector<int8_t>{1}}})
            .expected({{10}, element::i8, std::vector<int8_t>{0, 0, 2, -2, 6, -6, 4, -4, -128, -128}}),
    };
    return bitwiseParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_BitwiseLeftShift_With_Hardcoded_Refs,
                         ReferenceBitwiseLayerTest,
                         ::testing::ValuesIn(generateBitwiseParams()),
                         ReferenceBitwiseLayerTest::getTestCaseName);

}  // namespace
}  // namespace BitwiseOpsRefTestDefinitions
}  // namespace reference_tests
