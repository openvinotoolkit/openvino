// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/parameter_result.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::I32
    };

    INSTANTIATE_TEST_CASE_P(smoke_Check, ParameterResultSubgraphTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(std::vector<size_t>({20, 10, 10, 10})),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ParameterResultSubgraphTest::getTestCaseName);
}  // namespace

