// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/eltwise_reshape_activation.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;
namespace {
const std::vector<std::vector<std::vector<size_t>>> shapes = {{{1, 64}, {64, 1}},
                                                              {{8, 256}, {16, 128}},
                                                              {{6, 384}, {18, 128}},
                                                              {{8, 2048}, {32, 512}},
                                                              {{2, 4, 64, 64}, {1, 8, 64, 64}}};

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

std::vector<std::map<std::string, std::string>> additional_config = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                     {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseReshapeActivationTest,
                         EltwiseReshapeActivation,
                         ::testing::Combine(::testing::ValuesIn(shapes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(additional_config)),
                         EltwiseReshapeActivation::getTestCaseName);

}  // namespace
