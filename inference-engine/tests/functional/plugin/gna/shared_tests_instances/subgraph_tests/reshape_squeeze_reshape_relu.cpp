// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include <vector>
#include "subgraph_tests/reshape_squeeze_reshape_relu.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    std::vector<SqueezeShape> inputs{
            {{1, 1, 3}, {0, 1}},
            {{1, 1, 3}, {0}},
            {{1, 1, 3}, {1}},
            {{1, 3, 1}, {0, 2}},
            {{1, 3, 1}, {0}},
            {{1, 3, 1}, {2}},
            {{3, 1, 1}, {1, 2}},
            {{3, 1, 1}, {1}},
            {{3, 1, 1}, {2}},
            {{4, 1, 3, 1}, {1, 3}},
            {{4, 1, 1, 3}, {1, 2}},
            {{1, 4, 1, 3}, {0, 2}},
            {{1, 3, 5, 2, 1}, {0, 4}},
            {{3, 1, 2, 4, 4, 3}, {1}},
            {{1, 1, 1, 1, 1, 3}, {0, 1, 2, 3, 4}},
            {{1, 1, 1, 1, 1, 3}, {1, 3}},
            {{1}, {0}},
    };

    std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                             InferenceEngine::Precision::FP16,
    };

    INSTANTIATE_TEST_CASE_P(reshape_squeeze_reshape_relu, ReshapeSqueezeReshapeRelu,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputs),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::Values(true)),
                            ReshapeSqueezeReshapeRelu::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(reshape_unsqueeze_reshape_relu, ReshapeSqueezeReshapeRelu,
                            ::testing::Combine(
                                    ::testing::ValuesIn(inputs),
                                    ::testing::ValuesIn(netPrecisions),
                                    ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                    ::testing::Values(false)),
                            ReshapeSqueezeReshapeRelu::getTestCaseName);
}  // namespace
