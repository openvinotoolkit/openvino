// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/activation_fq.hpp"

#include <gna/gna_config.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP16,
                                                               InferenceEngine::Precision::FP32};

using ConfigType = std::map<std::string, std::string>;
const ConfigType configFP32 = {
    {"GNA_DEVICE_MODE", "GNA_SW_FP32"},
};

const ConfigType configSWExact = {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "NO"}};

const std::vector<std::pair<std::string, ConfigType>> gnaQuantModes = {{"sw_fp32", configFP32},
                                                                       {"sw_exact", configSWExact}};

const std::vector<std::vector<size_t>> inputShapes = {{1, 250}, {1, 640}, {1, 1024}};

const std::vector<size_t> level = {65535};
const std::vector<std::vector<float>> inputParams = {{-1, 1, 0.01}, {-5, 5, 1}, {-100, 100, 1}, {-16, 16, 1}};

const std::vector<std::vector<std::vector<size_t>>> constShapes = {{{1}}};

const auto fqParams =
    ::testing::Combine(::testing::Values(level), ::testing::ValuesIn(constShapes), ::testing::ValuesIn(inputParams));

const std::vector<ngraph::helpers::ActivationTypes> activations = {ngraph::helpers::ActivationTypes::Sigmoid,
                                                                   ngraph::helpers::ActivationTypes::Tanh,
                                                                   ngraph::helpers::ActivationTypes::Relu,
                                                                   ngraph::helpers::ActivationTypes::Log,
                                                                   ngraph::helpers::ActivationTypes::Abs,
                                                                   ngraph::helpers::ActivationTypes::Sign,
                                                                   ngraph::helpers::ActivationTypes::Exp};

INSTANTIATE_TEST_SUITE_P(smoke_ActivationFQSubgraph,
                         ActivationFakeQuantizeSubgraphTest,
                         ::testing::Combine(fqParams,
                                            ::testing::ValuesIn(activations),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(gnaQuantModes)),
                         ActivationFakeQuantizeSubgraphTest::getTestCaseName);
}  // namespace
