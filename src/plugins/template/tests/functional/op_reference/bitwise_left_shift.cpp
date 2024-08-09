// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_left_shift.hpp"

#include <gtest/gtest.h>

#include <limits>

#include "bitwise.hpp"

using namespace ov;

namespace reference_tests {
namespace BitwiseOpsRefTestDefinitions {
namespace {

// Off clang format to avoid long single column vectors

// clang-format off
std::vector<RefBitwiseParams> generateBitwiseParams() {
    constexpr int32_t min_int32 = std::numeric_limits<int32_t>::min();
    std::vector<RefBitwiseParams> bitwiseParams{
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{10}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1}, element::i32, std::vector<int32_t>{1}}})
            .expected({{10}, element::i32, std::vector<int32_t>{0, 0, 2, -2, 6, -6, 4, -4, 128, -128}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 5}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1, 1}, element::i32, std::vector<int32_t>{1}}})
            .expected({{2, 5}, element::i32, std::vector<int32_t>{0, 0, 2, -2, 6, -6, 4, -4, 128, -128}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 5}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{2, 1}, element::i32, std::vector<int32_t>{1, 2}}})
            .expected({{2, 5}, element::i32, std::vector<int32_t>{0, 0, 2, -2, 6, -12, 8, -8, 256, -256}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{10}, element::i8, std::vector<int8_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{1}, element::i8, std::vector<int8_t>{1}}})
            .expected({{10}, element::i8, std::vector<int8_t>{0, 0, 2, -2, 6, -6, 4, -4, -128, -128}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 5}, element::i64, std::vector<int64_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i64, std::vector<int64_t>{0, 1, 2, 3, 8, 31}}})
            .expected({{6, 2, 5},
                       element::i64,
                       std::vector<int64_t>{0,           0,          1,           -1,           3,
                                            -3,          2,          -2,          64,           -64,
                                            0,           0,          2,           -2,           6,
                                            -6,          4,          -4,          128,          -128,
                                            0,           0,          4,           -4,           12,
                                            -12,         8,          -8,          256,          -256,
                                            0,           0,          8,           -8,           24,
                                            -24,         16,         -16,         512,          -512,
                                            0,           0,          256,         -256,         768,
                                            -768,        512,        -512,        16384,        -16384,
                                            0,           0,          2147483648L,  min_int32,  6442450944L,
                                            -6442450944L, 4294967296L, -4294967296L, 137438953472L, -137438953472L}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 5}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i32, std::vector<int32_t>{0, 1, 2, 3, 8, 31}}})
            .expected(
                {{6, 2, 5},
                 element::i32,
                 std::vector<int32_t>{
                     0,     0,      1,   -1,   3,           -3,          2,           -2,          64,  -64,  0,   0,
                     2,     -2,     6,   -6,   4,           -4,          128,         -128,        0,   0,    4,   -4,
                     12,    -12,    8,   -8,   256,         -256,        0,           0,           8,   -8,   24,  -24,
                     16,    -16,    512, -512, 0,           0,           256,         -256,        768, -768, 512, -512,
                     16384, -16384, 0, 0, min_int32, min_int32, min_int32, min_int32, 0, 0, 0, 0}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 5}, element::i16, std::vector<int16_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i16, std::vector<int16_t>{0, 1, 2, 3, 8, 15}}})
            .expected({{6, 2, 5},
                       element::i16,
                       std::vector<int16_t>{
                           0,    0,   1,    -1,    3,      -3,  2,  -2,     64,     -64,    0,      0, 2,   -2,   6,
                           -6,   4,   -4,   128,   -128,   0,   0,  4,      -4,     12,     -12,    8, -8,  256,  -256,
                           0,    0,   8,    -8,    24,     -24, 16, -16,    512,    -512,   0,      0, 256, -256, 768,
                           -768, 512, -512, 16384, -16384, 0,   0,  -32768, -32768, -32768, -32768, 0, 0,   0,    0}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 5}, element::i8, std::vector<int8_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
                     {{6, 1, 1}, element::i8, std::vector<int8_t>{0, 1, 2, 3, 4, 7}}})
            .expected({{6, 2, 5},
                       element::i8,
                       std::vector<int8_t>{0,   0,  1,   -1,   3,    -3,  2,  -2,   64,   -64,  0,    0, 2,  -2,  6,
                                           -6,  4,  -4,  -128, -128, 0,   0,  4,    -4,   12,   -12,  8, -8, 0,   0,
                                           0,   0,  8,   -8,   24,   -24, 16, -16,  0,    0,    0,    0, 16, -16, 48,
                                           -48, 32, -32, 0,    0,    0,   0,  -128, -128, -128, -128, 0, 0,  0,   0}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 6}, element::u8, std::vector<uint8_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u8, std::vector<uint8_t>{0, 1, 2, 3, 4, 7}}})
            .expected({{6, 2, 6},
                       element::u8,
                       std::vector<uint8_t>{0,  1,   2,  3,   4,   5,   6,  7,   8,   9,   51,  64,  0,   2,  4,
                                            6,  8,   10, 12,  14,  16,  18, 102, 128, 0,   4,   8,   12,  16, 20,
                                            24, 28,  32, 36,  204, 0,   0,  8,   16,  24,  32,  40,  48,  56, 64,
                                            72, 152, 0,  0,   16,  32,  48, 64,  80,  96,  112, 128, 144, 48, 0,
                                            0,  128, 0,  128, 0,   128, 0,  128, 0,   128, 128, 0}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 6}, element::u16, std::vector<uint16_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u16, std::vector<uint16_t>{0, 1, 2, 3, 4, 7}}})
            .expected(
                {{6, 2, 6},
                 element::u16,
                 std::vector<uint16_t>{0,  1,   2,   3,   4,   5,   6,   7,   8,    9,    51,   64,  0,   2,   4,
                                       6,  8,   10,  12,  14,  16,  18,  102, 128,  0,    4,    8,   12,  16,  20,
                                       24, 28,  32,  36,  204, 256, 0,   8,   16,   24,   32,   40,  48,  56,  64,
                                       72, 408, 512, 0,   16,  32,  48,  64,  80,   96,   112,  128, 144, 816, 1024,
                                       0,  128, 256, 384, 512, 640, 768, 896, 1024, 1152, 6528, 8192}}),

        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 6}, element::u32, std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u32, std::vector<uint32_t>{0, 1, 2, 3, 4, 7}}})
            .expected(
                {{6, 2, 6},
                 element::u32,
                 std::vector<uint32_t>{0,  1,   2,   3,   4,   5,   6,   7,   8,    9,    51,   64,  0,   2,   4,
                                       6,  8,   10,  12,  14,  16,  18,  102, 128,  0,    4,    8,   12,  16,  20,
                                       24, 28,  32,  36,  204, 256, 0,   8,   16,   24,   32,   40,  48,  56,  64,
                                       72, 408, 512, 0,   16,  32,  48,  64,  80,   96,   112,  128, 144, 816, 1024,
                                       0,  128, 256, 384, 512, 640, 768, 896, 1024, 1152, 6528, 8192}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
            .inputs({{{2, 6}, element::u64, std::vector<uint64_t>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 51, 64}},
                     {{6, 1, 1}, element::u64, std::vector<uint64_t>{0, 1, 2, 3, 8, 63}}})
            .expected({{6, 2, 6},
                       element::u64,
                       std::vector<uint64_t>{
                           0,  1,   2,   3,   4,    5,    6,    7,    8,    9,    51,    64,    0,  2,  4,   6,
                           8,  10,  12,  14,  16,   18,   102,  128,  0,    4,    8,     12,    16, 20, 24,  28,
                           32, 36,  204, 256, 0,    8,    16,   24,   32,   40,   48,    56,    64, 72, 408, 512,
                           0,  256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 13056, 16384, 0, 9223372036854775808ul,
                           0, 9223372036854775808ul, 0, 9223372036854775808ul, 0, 9223372036854775808ul, 0, 9223372036854775808ul,
                           9223372036854775808ul,  0}}),

        //// Note: Negative shift, Implementation defined or undefined behavior
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_LEFT_SHIFT)
        //     .inputs({{{1, 10}, element::i32, std::vector<int32_t>{0, 0, 1, -1, 3, -3, 2, -2, 64, -64}},
        //              {{1}, element::i32, std::vector<int32_t>{-1}}})
        //     .expected({{1, 10}, element::i32, std::vector<int32_t>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}}),
    };
    return bitwiseParams;
}
// clang-format on

INSTANTIATE_TEST_SUITE_P(smoke_BitwiseLeftShift_With_Hardcoded_Refs,
                         ReferenceBitwiseLayerTest,
                         ::testing::ValuesIn(generateBitwiseParams()),
                         ReferenceBitwiseLayerTest::getTestCaseName);

}  // namespace
}  // namespace BitwiseOpsRefTestDefinitions
}  // namespace reference_tests
