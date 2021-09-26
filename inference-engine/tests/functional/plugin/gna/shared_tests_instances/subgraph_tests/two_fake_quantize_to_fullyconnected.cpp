// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <gna/gna_config.hpp>

#include "subgraph_tests/two_fake_quantize_to_fullyconnected.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
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
const std::vector<std::vector<std::vector<size_t>>> constShapes = {
        {{1}, {1024, 1}}
};

const std::vector<std::vector<std::vector<size_t>>> constShapes_int16 = {
        {{1}, {1}}
};

const std::vector<size_t> levels_fp = {255, 65535};
const std::vector<std::vector<size_t>> levels_i16 = {{65535, 65535}, {32767, 32767}, {16383, 16383}};
const std::vector<std::vector<size_t>> levels_i8 = {{255, 255}};

const std::vector<std::vector<float>> fqArgs = {{-2.0f, 2.0f, -2.0f, 2.0f}};
const std::vector<std::vector<float>> inputParams = {{-64, 64, 1}, {-10, 10, 0.1}};
const std::vector<std::vector<float>> inputParams_I8 = {{-2.0f, 2.0f, 0.1f}};

const std::vector<bool> biases = {false, true};

const auto fqParams = ::testing::Combine(
        ::testing::Values(levels_fp),
        ::testing::ValuesIn(constShapes),
        ::testing::ValuesIn(fqArgs),
        ::testing::ValuesIn(inputParams)
);

const auto fqParams_I8 = ::testing::Combine(
        ::testing::ValuesIn(levels_i8),
        ::testing::ValuesIn(constShapes),
        ::testing::ValuesIn(fqArgs),
        ::testing::ValuesIn(inputParams_I8)
);

const auto fqParams_I16 = ::testing::Combine(
        ::testing::ValuesIn(levels_i16),
        ::testing::ValuesIn(constShapes_int16),
        ::testing::ValuesIn(fqArgs),
        ::testing::ValuesIn(inputParams_I8)
);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_subgraph, FakeQuantizeSubgraphTest,
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

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_subgraph_U8, FakeQuantizeSubgraphTest,
                        ::testing::Combine(
                                fqParams_I8,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapes),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(gnaQuantModes_I8),
                                ::testing::ValuesIn(biases)),
                        FakeQuantizeSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_FakeQuantize_subgraph_I16, FakeQuantizeSubgraphTest,
                        ::testing::Combine(
                                fqParams_I16,
                                ::testing::ValuesIn(netPrecisions),
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
