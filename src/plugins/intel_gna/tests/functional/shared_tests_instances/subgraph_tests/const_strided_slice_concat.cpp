// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/const_strided_slice_concat.hpp"

#include <cstdint>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

std::vector<uint32_t> inputChunksSizes = {32, 64};
std::vector<uint32_t> inputChunksNumber = {4, 7};

std::vector<uint32_t> constChunksSizes = {96, 128};
std::vector<uint32_t> constChunksNumber = {1, 3};

INSTANTIATE_TEST_SUITE_P(smoke_ConstStridedSliceConcatTest,
                         ConstStridedSliceConcatTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs),
                                            ::testing::ValuesIn(inputChunksSizes),
                                            ::testing::ValuesIn(inputChunksNumber),
                                            ::testing::ValuesIn(constChunksSizes),
                                            ::testing::ValuesIn(constChunksNumber)),
                         ConstStridedSliceConcatTest::getTestCaseName);
}  // namespace
