// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/relu_shape_of.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::I32
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Check, ReluShapeOfSubgraphTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(std::vector<size_t>({20, 10, 10, 10})),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            ReluShapeOfSubgraphTest::getTestCaseName);
}  // namespace
