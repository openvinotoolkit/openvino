// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/quantized_mat_mul.hpp"

using namespace LayerTestsDefinitions;
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

const std::vector<size_t> levels = {256};
const std::vector<QuantizationGranularity> granularity = {Pertensor};

const auto quantParams_i8i8 = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity),
        ::testing::Values(InferenceEngine::Precision::I8)
);

const auto quantParams_u8i8 = ::testing::Combine(
        ::testing::ValuesIn(levels),
        ::testing::ValuesIn(granularity),
        ::testing::Values(InferenceEngine::Precision::U8)
);

INSTANTIATE_TEST_CASE_P(smoke_QuantMatMul_i8i8, QuantMatMulTest,
                        ::testing::Combine(
                                quantParams_i8i8,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(shapesA),
                                ::testing::ValuesIn(shapesB),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        QuantMatMulTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_QuantMatMul_u8i8, QuantMatMulTest,
                        ::testing::Combine(
                                quantParams_u8i8,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(shapesA),
                                ::testing::ValuesIn(shapesB),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        QuantMatMulTest::getTestCaseName);

} // namespace

