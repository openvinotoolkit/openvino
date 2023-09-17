// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "subgraph_tests/reshape_permute_reshape.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<std::vector<std::vector<size_t>>> inputs{
    {{1, 4, 160}, {0, 2, 1}},
    {{1, 160, 4}, {0, 2, 1}},
    {{8, 16}, {1, 0}},
    {{1, 1, 4, 16}, {3, 1, 2, 0}},
    {{1, 8, 200}, {0, 2, 1}},
    {{1, 8, 16}, {2, 1, 0}},
};

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_reshape_permute_reshape,
                         ReshapePermuteReshape,
                         ::testing::Combine(::testing::ValuesIn(inputs),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA)),
                         ReshapePermuteReshape::getTestCaseName);
}  // namespace
