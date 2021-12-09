// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/crop4d.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

std::vector<StridedSliceSpecificParams> ss_only_test_cases = {
        StridedSliceSpecificParams{ { 1, 2, 100, 100 },
                                    { 0, 0, 0, 0 },
                                    { 1, 1, 1, 1 }, { 1, 1, 1, 1 },
                                    { 1, 1, 1, 1 }, { 1, 0, 1, 1 },  { 0, 0, 0, 0 },  { 1, 0, 0, 0 },  { 0, 0, 0, 0 } },
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {
        {
                {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
                {"GNA_COMPACT_MODE", "NO"}
        }
};

INSTANTIATE_TEST_SUITE_P(
        smoke_crop4d_gna, Crop4dTest,
        ::testing::Combine(
        ::testing::ValuesIn(ss_only_test_cases),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
        Crop4dTest::getTestCaseName);

}  // namespace