// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/concat_quantization_during_memory_requantization.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                         InferenceEngine::Precision::FP32};

std::vector<std::map<std::string, std::string>> additionalConfig = {
    {{"GNA_COMPACT_MODE", "NO"}},
    {{"GNA_COMPACT_MODE", "NO"}, {"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

std::vector<size_t> inputSizes = {128, 64, 32};

std::vector<size_t> hiddenSizes = {128, 64, 32};

INSTANTIATE_TEST_SUITE_P(smoke_concat_quant_memory_requant,
                         ConcatQuantDuringMemoryRequantTest,
                         ::testing::Combine(::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(inputSizes),
                                            ::testing::ValuesIn(hiddenSizes),
                                            ::testing::ValuesIn(additionalConfig)),
                         ConcatQuantDuringMemoryRequantTest::getTestCaseName);
}  // namespace
