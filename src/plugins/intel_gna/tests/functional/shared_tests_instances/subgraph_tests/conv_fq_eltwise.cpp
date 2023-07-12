// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/conv_fq_eltwise.hpp"

#include <gna/gna_config.hpp>
#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::map<std::string, std::string>> configs = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                 {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

const std::vector<std::vector<size_t>> inputShapes = {{1, 1024}};

const size_t levels = 65535;

const std::vector<std::vector<float>> inputParams = {{-10, 10, 1}};

const float convFQValue = 2.0f;

const auto fqParams =
    ::testing::Combine(::testing::Values(levels), ::testing::ValuesIn(inputParams), ::testing::Values(convFQValue));

const std::vector<std::vector<size_t>> kernels = {{1, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<size_t> inputChannels = {8};
const std::vector<size_t> outputChannels{4};

const auto convParams = ::testing::Combine(::testing::ValuesIn(kernels),
                                           ::testing::ValuesIn(strides),
                                           ::testing::ValuesIn(inputChannels),
                                           ::testing::ValuesIn(outputChannels));

INSTANTIATE_TEST_SUITE_P(smoke_ConvFqEltwiseTest,
                         ConvFqEltwiseTest,
                         ::testing::Combine(fqParams,
                                            convParams,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvFqEltwiseTest::getTestCaseName);

}  // namespace
