// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_right_shift.hpp"

#include <gtest/gtest.h>

#include "bitwise.hpp"

using namespace ov;

namespace reference_tests {
namespace BitwiseOpsRefTestDefinitions {
namespace {

std::vector<RefBitwiseParams> generateBitwiseParams() {
    std::vector<RefBitwiseParams> bitwiseParams{
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{10}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1}, element::i32, std::vector<int32_t>{1}}})
            .expected({{10}, element::i32, std::vector<int32_t>{0, 0, 0, -1, 1, -2, 1, -1, 32, -32}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{5, 2}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1, 1}, element::i32, std::vector<int32_t>{1}}})
            .expected({{5, 2}, element::i32, std::vector<int32_t>{0, 0, 0, -1, 1, -2, 1, -1, 32, -32}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 5}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{2, 1}, element::i32, std::vector<int32_t>{1, 2}}})
            .expected({{2, 5}, element::i32, std::vector<int32_t>{0, 0, 0, -1, 1, -1, 0, -1, 16, -16}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{10}, element::i8, std::vector<int8_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1}, element::i8, std::vector<int8_t>{1}}})
            .expected({{10}, element::i8, std::vector<int8_t>{0, 0, 0, -1, 1, -2, 1, -1, 32, -32}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 5}, element::i64, std::vector<int64_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i64, std::vector<int64_t>{0, 1, 2, 3, 8, 31}}})
            .expected({{6, 2, 5},
                       element::i64,
                       std::vector<int64_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64, 0, 0, 0, -1, 1, -2, 1, -1, 32, -32,
                                            0, 0, 0, -1, 0, -1, 0, -1, 16, -16, 0, 0, 0, -1, 0, -1, 0, -1, 8,  -8,
                                            0, 0, 0, -1, 0, -1, 0, -1, 0,  -1,  0, 0, 0, -1, 0, -1, 0, -1, 0,  -1}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 5}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i32, std::vector<int32_t>{0, 1, 2, 3, 8, 31}}})
            .expected({{6, 2, 5},
                       element::i32,
                       std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64, 0, 0, 0, -1, 1, -2, 1, -1, 32, -32,
                                            0, 0, 0, -1, 0, -1, 0, -1, 16, -16, 0, 0, 0, -1, 0, -1, 0, -1, 8,  -8,
                                            0, 0, 0, -1, 0, -1, 0, -1, 0,  -1,  0, 0, 0, -1, 0, -1, 0, -1, 0,  -1}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 5}, element::i16, std::vector<int16_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i16, std::vector<int16_t>{0, 1, 2, 3, 8, 16}}})
            .expected({{6, 2, 5},
                       element::i16,
                       std::vector<int16_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64, 0, 0, 0, -1, 1, -2, 1, -1, 32, -32,
                                            0, 0, 0, -1, 0, -1, 0, -1, 16, -16, 0, 0, 0, -1, 0, -1, 0, -1, 8,  -8,
                                            0, 0, 0, -1, 0, -1, 0, -1, 0,  -1,  0, 0, 0, -1, 0, -1, 0, -1, 0,  -1}}),

        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 5}, element::i8, std::vector<int8_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i8, std::vector<int8_t>{0, 1, 2, 3, 4, 7}}})
            .expected({{6, 2, 5},
                       element::i8,
                       std::vector<int8_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64, 0, 0, 0, -1, 1, -2, 1, -1, 32, -32,
                                           0, 0, 0, -1, 0, -1, 0, -1, 16, -16, 0, 0, 0, -1, 0, -1, 0, -1, 8,  -8,
                                           0, 0, 0, -1, 0, -1, 0, -1, 4,  -4,  0, 0, 0, -1, 0, -1, 0, -1, 0,  -1}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 6}, element::u8, std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u8, std::vector<uint8_t>{0, 1, 2, 3, 4, 7}}})
            .expected({{6, 2, 6}, element::u8, std::vector<uint8_t>{0, 1, 2, 3, 4,  5,  6, 7,  8,  9, 51, 64, 0, 0, 1,
                                                                    1, 2, 2, 3, 3,  4,  4, 25, 32, 0, 0,  0,  0, 1, 1,
                                                                    1, 1, 2, 2, 12, 16, 0, 0,  0,  0, 0,  0,  0, 0, 1,
                                                                    1, 6, 8, 0, 0,  0,  0, 0,  0,  0, 0,  0,  0, 3, 4,
                                                                    0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0,  0}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 6}, element::u16, std::vector<uint16_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u16, std::vector<uint16_t>{0, 1, 2, 3, 4, 7}}})
            .expected({{6, 2, 6}, element::u16, std::vector<uint16_t>{0, 1, 2, 3, 4,  5,  6, 7,  8,  9, 51, 64, 0, 0, 1,
                                                                      1, 2, 2, 3, 3,  4,  4, 25, 32, 0, 0,  0,  0, 1, 1,
                                                                      1, 1, 2, 2, 12, 16, 0, 0,  0,  0, 0,  0,  0, 0, 1,
                                                                      1, 6, 8, 0, 0,  0,  0, 0,  0,  0, 0,  0,  0, 3, 4,
                                                                      0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0,  0}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 6}, element::u32, std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u32, std::vector<uint32_t>{0, 1, 2, 3, 4, 7}}})
            .expected({{6, 2, 6}, element::u32, std::vector<uint32_t>{0, 1, 2, 3, 4,  5,  6, 7,  8,  9, 51, 64, 0, 0, 1,
                                                                      1, 2, 2, 3, 3,  4,  4, 25, 32, 0, 0,  0,  0, 1, 1,
                                                                      1, 1, 2, 2, 12, 16, 0, 0,  0,  0, 0,  0,  0, 0, 1,
                                                                      1, 6, 8, 0, 0,  0,  0, 0,  0,  0, 0,  0,  0, 3, 4,
                                                                      0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0,  0}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
            .inputs({{{2, 6}, element::u64, std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u64, std::vector<uint64_t>{0, 1, 2, 3, 8, 63}}})
            .expected({{6, 2, 6}, element::u64, std::vector<uint64_t>{0, 1, 2, 3, 4,  5,  6, 7,  8,  9, 51, 64, 0, 0, 1,
                                                                      1, 2, 2, 3, 3,  4,  4, 25, 32, 0, 0,  0,  0, 1, 1,
                                                                      1, 1, 2, 2, 12, 16, 0, 0,  0,  0, 0,  0,  0, 0, 1,
                                                                      1, 6, 8, 0, 0,  0,  0, 0,  0,  0, 0,  0,  0, 0, 0,
                                                                      0, 0, 0, 0, 0,  0,  0, 0,  0,  0, 0,  0}}),

        //// Note: Negative shift, Implementation defined or undefined behavior
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_RIGHT_SHIFT)
        //     .inputs({{{1, 10}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
        //              {{1}, element::i32, std::vector<int32_t>{-1}}})
        //     .expected({{1, 10}, element::i32, std::vector<int32_t>{0, 0, 0, -1, 0, -1, 0, -1, 0, -1}}),
    };
    return bitwiseParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_BitwiseRightShift_With_Hardcoded_Refs,
                         ReferenceBitwiseLayerTest,
                         ::testing::ValuesIn(generateBitwiseParams()),
                         ReferenceBitwiseLayerTest::getTestCaseName);

}  // namespace
}  // namespace BitwiseOpsRefTestDefinitions
}  // namespace reference_tests
