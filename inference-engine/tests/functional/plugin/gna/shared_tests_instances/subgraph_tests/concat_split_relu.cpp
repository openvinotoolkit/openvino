// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <vector>
#include "subgraph_tests/concat_split_relu.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16,
    };

    std::map<std::string, std::string> additional_config = {
    };

    INSTANTIATE_TEST_CASE_P(split_connected, ConcatSplitRelu,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::Values(additional_config)),
            ConcatSplitRelu::getTestCaseName);
}  // namespace
