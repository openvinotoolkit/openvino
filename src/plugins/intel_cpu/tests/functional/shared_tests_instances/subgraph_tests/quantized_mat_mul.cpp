// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <limits>

#include "subgraph_tests/quantized_mat_mul.hpp"

using namespace SubgraphTestsDefinitions;
using namespace ngraph::helpers;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<std::vector<size_t>> shapesA = {
        {1, 4, 5, 6}
};

const std::vector<std::vector<size_t>> shapesB = {
        {1, 4, 6, 4}
};

const std::vector<QuantRange> ranges_i8 = {
        { -127, 128 }
};

const std::vector<QuantRange> ranges_u8 = {
        { 0, 255 }
};

const std::vector<QuantRange> ranges_i16 = {
        { -32768, 32767 }
};

const std::vector<QuantRange> ranges_i32 = {
        { INT32_MIN, INT32_MAX }
};

const std::vector<uint64_t> levels_8 = {256};
const std::vector<uint64_t> levels_16 = {65536};
const std::vector<uint64_t> levels_32 = {4294967296};
const std::vector<QuantizationGranularity> granularity = {Pertensor};

const auto quantParams_i8 = ::testing::Combine(
        ::testing::ValuesIn(levels_8),
        ::testing::ValuesIn(ranges_u8),
        ::testing::ValuesIn(ranges_i8),
        ::testing::ValuesIn(granularity),
        ::testing::Values(InferenceEngine::Precision::I8)
);

const auto quantParams_u8 = ::testing::Combine(
        ::testing::ValuesIn(levels_8),
        ::testing::ValuesIn(ranges_u8),
        ::testing::ValuesIn(ranges_u8),
        ::testing::ValuesIn(granularity),
        ::testing::Values(InferenceEngine::Precision::U8)
);

const auto quantParams_i16 = ::testing::Combine(
        ::testing::ValuesIn(levels_16),
        ::testing::ValuesIn(ranges_i32),
        ::testing::ValuesIn(ranges_i16),
        ::testing::ValuesIn(granularity),
        ::testing::Values(InferenceEngine::Precision::I16)
);

const auto quantParams_i32 = ::testing::Combine(
        ::testing::ValuesIn(levels_32),
        ::testing::ValuesIn(ranges_i32),
        ::testing::ValuesIn(ranges_i32),
        ::testing::ValuesIn(granularity),
        ::testing::Values(InferenceEngine::Precision::I32)
);

INSTANTIATE_TEST_SUITE_P(smoke_QuantMatMul_i8i8, QuantMatMulTest,
                        ::testing::Combine(
                                quantParams_i8,
                                quantParams_i8,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(shapesA),
                                ::testing::ValuesIn(shapesB),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        QuantMatMulTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_QuantMatMul_u8i8, QuantMatMulTest,
                        ::testing::Combine(
                                quantParams_u8,
                                quantParams_i8,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(shapesA),
                                ::testing::ValuesIn(shapesB),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        QuantMatMulTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_QuantMatMul_i16i32, QuantMatMulTest,
                        ::testing::Combine(
                                quantParams_i16,
                                quantParams_i32,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(shapesA),
                                ::testing::ValuesIn(shapesB),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        QuantMatMulTest::getTestCaseName);

} // namespace

