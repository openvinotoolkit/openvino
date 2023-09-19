// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/bitwise_not.hpp"
#include "bitwise.hpp"

using namespace ov;

namespace reference_tests {
namespace BitwiseOpsRefTestDefinitions {
namespace {

std::vector<RefBitwiseParams> generateBitwiseParams() {
    std::vector<RefBitwiseParams> bitwiseParams {
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{2, 2}, element::boolean, std::vector<char> {true, false, true, false}}})
            .expected({{2, 2}, element::boolean, std::vector<char> {false, true, false, true}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{7}, element::i8, std::vector<int8_t> {127, -128, 0, 63, -64, 6, -7}}})
            .expected({{7}, element::i8, std::vector<int8_t> {-128, 127, -1, -64, 63, -7, 6}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{7}, element::u8, std::vector<uint8_t> {255, 0, 127, 255, 127, 134, 120}}})
            .expected({{7}, element::u8, std::vector<uint8_t> {0, 255, 128, 0, 128, 121, 135}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{7}, element::i16, std::vector<int16_t> {32767, -32768, 0, 16383, -16384, 6, -7}}})
            .expected({{7}, element::i16, std::vector<int16_t> {-32768, 32767, -1, -16384, 16383, -7, 6}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{7}, element::u16, std::vector<uint16_t> {65535, 0, 32767, 65535, 32767, 32774, 32760}}})
            .expected({{7}, element::u16, std::vector<uint16_t> {0, 65535, 32768, 0, 32768, 32761, 32775}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{7}, element::i32, std::vector<int32_t> {2147483647, -2147483648, 0, 1073741823, -1073741824, 6, -7}}})
            .expected({{7}, element::i32, std::vector<int32_t> {-2147483648, 2147483647, -1, -1073741824, 1073741823, -7, 6}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{7}, element::u32, std::vector<uint32_t> {4294967295, 0, 2147483647, 4294967295, 2147483647, 2147483654, 2147483640}}})
            .expected({{7}, element::u32, std::vector<uint32_t> {0, 4294967295, 2147483648, 0, 2147483648, 2147483641, 2147483655}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{7}, element::i64, std::vector<int64_t> {9223372036854775807, -9223372036854775808, 0, 4611686018427387904, -4611686018427387904, 6, -7}}})
            .expected({{7}, element::i64, std::vector<int64_t> {-9223372036854775808, 9223372036854775807, -1, -4611686018427387905, 4611686018427387903, -7, 6}}),
        Builder {}
            .opType(BitwiseTypes::BITWISE_NOT)
            .inputs({{{3}, element::u64, std::vector<uint64_t> {std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::min(), 5}}})
            .expected({{3}, element::u64, std::vector<uint64_t> {std::numeric_limits<uint64_t>::min(), std::numeric_limits<uint64_t>::max(), std::numeric_limits<uint64_t>::max()-5}}),
            };
    return bitwiseParams;
}


INSTANTIATE_TEST_SUITE_P(smoke_BitwiseNot_With_Hardcoded_Refs, ReferenceBitwiseLayerTest, ::testing::ValuesIn(generateBitwiseParams()),
                         ReferenceBitwiseLayerTest::getTestCaseName);

}  // namespace
}  // namespace BitwiseOpsRefTestDefinitions
}  // namespace reference_tests
