// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gna/gna_config.hpp>

#include "subgraph_tests/two_fake_quantize_to_fullyconnected.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
};

const std::vector<InferenceEngine::Precision> netPrecisions_I8 = {
        InferenceEngine::Precision::I8,
};

using ConfigType = std::map<std::string, std::string>;
const ConfigType configFP32 = {
        {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
};

const ConfigType configSWExact = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_COMPACT_MODE", "NO"}
};


/**
 * @brief specific quantisation mode to be used internally
 */
const std::vector<std::pair<std::string, ConfigType>> gnaQuantModes = {
            {"sw_fp32", configFP32},
    };

const std::vector<std::pair<std::string, ConfigType>> gnaQuantModes_I8 = {
        {"gna_sw_exact", configSWExact},
};

const std::vector<std::vector<size_t>> inputShapes = {
        {1, 440}
};
const std::vector<std::vector<size_t>> constShapes = {{1}};
const std::vector<size_t> levels = {255, 65535};

const std::vector<std::vector<float>> fqArgs = {{0, 10, 2, 6}};
const std::vector<std::vector<float>> inputParams = {{-1000, 1000, 10.0}, {-10, 10, 0.1}, {}};
const std::vector<std::vector<float>> inputParams_I8 = {{-100, 100, 10}, {-10, 10, 1.0}, {}};


const std::vector<bool> biases = {true, false};

const auto fqParams = ::testing::Combine(
        ::testing::Values(levels),
        ::testing::ValuesIn(constShapes),
        ::testing::ValuesIn(fqArgs),
        ::testing::ValuesIn(inputParams)
);

const auto fqParams_I8 = ::testing::Combine(
        ::testing::Values(levels),
        ::testing::ValuesIn(constShapes),
        ::testing::ValuesIn(fqArgs),
        ::testing::ValuesIn(inputParams_I8)
);

INSTANTIATE_TEST_CASE_P(DISABLED_smoke_FakeQuantize_subgraph, FakeQuantizeSubgraphTest,
                        ::testing::Combine(
                                fqParams,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(gnaQuantModes),
                                ::testing::ValuesIn(biases)),
                        FakeQuantizeSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_smoke_FakeQuantize_subgraph_I8, FakeQuantizeSubgraphTest,
                        ::testing::Combine(
                                fqParams_I8,
                                ::testing::ValuesIn(netPrecisions_I8),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(gnaQuantModes_I8),
                                ::testing::ValuesIn(biases)),
                        FakeQuantizeSubgraphTest::getTestCaseName);

}  // namespace
