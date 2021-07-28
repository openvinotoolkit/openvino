// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "subgraph_tests/delayed_copy_layer.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
    };

    std::map<std::string, std::string> additional_config = {
            {"GNA_COMPACT_MODE", "NO"}
    };

    INSTANTIATE_TEST_SUITE_P(delayed_copy_layer, DelayedCopyTest,
                            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(CommonTestUtils::DEVICE_GNA),
            ::testing::Values(additional_config)),
                            DelayedCopyTest::getTestCaseName);
}  // namespace
