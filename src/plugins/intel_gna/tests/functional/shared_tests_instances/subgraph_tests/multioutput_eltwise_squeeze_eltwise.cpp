// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/multioutput_eltwise_squeeze_eltwise.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<std::vector<std::vector<size_t>>> inputs{{{1, 16}},
                                                     {{2, 16}},
                                                     {{1, 160}},
                                                     {{8, 40}},
                                                     {{3, 8}},
                                                     {{4, 32}},
                                                     {{5, 64}},
                                                     {{6, 128}},
                                                     {{7, 256}},
                                                     {{8, 512}},
                                                     {{8, 1024}}};

std::vector<std::map<std::string, std::string>> configs = {{{"GNA_COMPACT_MODE", "NO"}},
                                                           {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_multioutput_eltwise_identity,
                         MultioutputEltwiseReshapeEltwise,
                         ::testing::Combine(::testing::ValuesIn(inputs),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         MultioutputEltwiseReshapeEltwise::getTestCaseName);
}  // namespace
