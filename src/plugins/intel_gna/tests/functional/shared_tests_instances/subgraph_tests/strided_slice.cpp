// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/strided_slice.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<StridedSliceSpecificParams> ss_only_test_cases = {
    StridedSliceSpecificParams{{1, 2, 100, 100},
                               {0, 0, 0, 0},
                               {1, 1, 1, 1},
                               {1, 1, 1, 1},
                               {1, 1, 1, 1},
                               {1, 0, 1, 1},
                               {0, 0, 0, 0},
                               {1, 0, 0, 0},
                               {0, 0, 0, 0}},
    StridedSliceSpecificParams{{1, 2, 100},
                               {0, 0, 0},
                               {1, 1, 1},
                               {1, 1, 1},
                               {1, 1, 1},
                               {1, 0, 1},
                               {0, 0, 0},
                               {1, 0, 0},
                               {0, 0, 0}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_COMPACT_MODE", "NO"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}}};

INSTANTIATE_TEST_SUITE_P(smoke_stridedslice_gna,
                         StridedSliceTest,
                         ::testing::Combine(::testing::ValuesIn(ss_only_test_cases),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         StridedSliceTest::getTestCaseName);

}  // namespace
