// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/bitwise_and.hpp"

#include <gtest/gtest.h>

#include "bitwise.hpp"

using namespace ov;

namespace reference_tests {
namespace BitwiseOpsRefTestDefinitions {
namespace {

std::vector<RefBitwiseParams> generateBitwiseParams() {
    std::vector<RefBitwiseParams> bitwiseParams{
        Builder{}
            .opType(BitwiseTypes::BITWISE_AND)
            .inputs({{{2, 2}, element::boolean, std::vector<char>{true, false, true, false}},
                     {{2, 2}, element::boolean, std::vector<char>{true, false, false, true}}})
            .expected({{2, 2}, element::boolean, std::vector<char>{true, false, false, false}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_AND)
            .inputs(
                {{{3}, element::i8, std::vector<int8_t>{125, 2, 3}}, {{3}, element::i8, std::vector<int8_t>{-5, 2, 1}}})
            .expected({{3}, element::i8, std::vector<int8_t>{121, 2, 1}}),
        Builder{}
            .opType(BitwiseTypes::BITWISE_AND)
            .inputs({{{3}, element::u8, std::vector<uint8_t>{125, 2, 3}},
                     {{3}, element::u8, std::vector<uint8_t>{5, 2, 1}}})
            .expected({{3}, element::u8, std::vector<uint8_t>{5, 2, 1}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs({{{3},
        //               element::i8,
        //               std::vector<int8_t>{std::numeric_limits<int8_t>::max(), std::numeric_limits<int8_t>::min(),
        //               -7}}})
        //     .expected({{3},
        //                element::i8,
        //                std::vector<int8_t>{std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max(),
        //                6}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs(
        //         {{{3},
        //           element::u8,
        //           std::vector<uint8_t>{std::numeric_limits<uint8_t>::max(), std::numeric_limits<uint8_t>::min(),
        //           7}}})
        //     .expected({{3},
        //                element::u8,
        //                std::vector<uint8_t>{std::numeric_limits<uint8_t>::min(),
        //                                     std::numeric_limits<uint8_t>::max(),
        //                                     std::numeric_limits<uint8_t>::max() - 7}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs(
        //         {{{3},
        //           element::i16,
        //           std::vector<int16_t>{std::numeric_limits<int16_t>::max(), std::numeric_limits<int16_t>::min(),
        //           -7}}})
        //     .expected(
        //         {{3},
        //          element::i16,
        //          std::vector<int16_t>{std::numeric_limits<int16_t>::min(), std::numeric_limits<int16_t>::max(), 6}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs({{{3},
        //               element::u16,
        //               std::vector<uint16_t>{std::numeric_limits<uint16_t>::max(),
        //                                     std::numeric_limits<uint16_t>::min(),
        //                                     7}}})
        //     .expected({{3},
        //                element::u16,
        //                std::vector<uint16_t>{std::numeric_limits<uint16_t>::min(),
        //                                      std::numeric_limits<uint16_t>::max(),
        //                                      std::numeric_limits<uint16_t>::max() - 7}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs(
        //         {{{3},
        //           element::i32,
        //           std::vector<int32_t>{std::numeric_limits<int32_t>::max(), std::numeric_limits<int32_t>::min(),
        //           -7}}})
        //     .expected(
        //         {{3},
        //          element::i32,
        //          std::vector<int32_t>{std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max(), 6}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs({{{3},
        //               element::u32,
        //               std::vector<uint32_t>{std::numeric_limits<uint32_t>::max(),
        //                                     std::numeric_limits<uint32_t>::min(),
        //                                     7}}})
        //     .expected({{3},
        //                element::u32,
        //                std::vector<uint32_t>{std::numeric_limits<uint32_t>::min(),
        //                                      std::numeric_limits<uint32_t>::max(),
        //                                      std::numeric_limits<uint32_t>::max() - 7}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs(
        //         {{{3},
        //           element::i64,
        //           std::vector<int64_t>{std::numeric_limits<int64_t>::max(), std::numeric_limits<int64_t>::min(),
        //           -7}}})
        //     .expected(
        //         {{3},
        //          element::i64,
        //          std::vector<int64_t>{std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max(), 6}}),
        // Builder{}
        //     .opType(BitwiseTypes::BITWISE_AND)
        //     .inputs({{{3},
        //               element::u64,
        //               std::vector<uint64_t>{std::numeric_limits<uint64_t>::max(),
        //                                     std::numeric_limits<uint64_t>::min(),
        //                                     7}}})
        //     .expected({{3},
        //                element::u64,
        //                std::vector<uint64_t>{std::numeric_limits<uint64_t>::min(),
        //                                      std::numeric_limits<uint64_t>::max(),
        //                                      std::numeric_limits<uint64_t>::max() - 7}}),
    };
    return bitwiseParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_BitwiseAnd_With_Hardcoded_Refs,
                         ReferenceBitwiseLayerTest,
                         ::testing::ValuesIn(generateBitwiseParams()),
                         ReferenceBitwiseLayerTest::getTestCaseName);

}  // namespace
}  // namespace BitwiseOpsRefTestDefinitions
}  // namespace reference_tests
