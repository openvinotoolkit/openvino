// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gna/gna_config.hpp>

#include "subgraph_tests/clamp_fq.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32
};


using ConfigType = std::map<std::string, std::string>;
const ConfigType configFP32 = {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
};

const ConfigType configSWExact = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_COMPACT_MODE", "NO"}
};

const std::vector<std::pair<std::string, ConfigType>> gnaQuantModes = {
        {"sw_fp32", configFP32},
        {"sw_exact", configSWExact}
};

const std::vector<std::vector<size_t>> inputShapes = {
        {1, 250},
        {1, 640},
        {1, 1024}
};

const std::vector<size_t> level = {65535};
const std::vector<std::vector<float>> inputParams = {
                                                     {-16, 16, 1},
                                                     {-40, 40, 1},
                                                     {0, 10, 1},
                                                     {-50, 0, 1},
                                                     {0, 50, 1},
                                                     {-100, 100, 1},
                                                     {-5, 5, 1},
                                                     {-10, 10, 1},
                                                     {-14, 14, 1}
};
const std::vector<std::vector<float>> clampMaxMin = {
                                                      {-5, 5},
                                                      {-20, 20},
                                                      {-25, 0},
                                                      {-30, 25},
                                                      {0, 40},
                                                      {-10, 10},
                                                      {-14, 14}
};

const std::vector<std::vector<std::vector<size_t>>> constShapes = {
        {{1}}
};

const auto fqParams = ::testing::Combine(
        ::testing::Values(level),
        ::testing::ValuesIn(constShapes),
        ::testing::ValuesIn(clampMaxMin),
        ::testing::ValuesIn(inputParams)
);


INSTANTIATE_TEST_SUITE_P(smoke_Clamp_FQ_subgraph, ClampFakeQuantizeSubgraphTest,
                        ::testing::Combine(
                                fqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(gnaQuantModes)),
                        ClampFakeQuantizeSubgraphTest::getTestCaseName);
} // namespace