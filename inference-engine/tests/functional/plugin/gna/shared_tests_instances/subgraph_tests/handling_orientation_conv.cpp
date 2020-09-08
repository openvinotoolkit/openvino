// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/handling_orientation_conv.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::map<std::string, std::string>> configs = {
        {
                {"GNA_SCALE_FACTOR_0", "1"},
                {"GNA_SCALE_FACTOR_1", "1"},
                {"GNA_COMPACT_MODE", "NO"},
        }
};

INSTANTIATE_TEST_CASE_P(handling_orientation, HandlingOrientationClass,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::ValuesIn(configs)),
                        HandlingOrientationClass::getTestCaseName);

