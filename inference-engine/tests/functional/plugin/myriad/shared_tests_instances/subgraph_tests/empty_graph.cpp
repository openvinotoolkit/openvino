// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "subgraph_tests/empty_graph.hpp"
#include "common_test_utils/test_constants.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(empty_graph, EmptyGraph,
                        ::testing::Combine(
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(std::vector<size_t >({1, 6, 40, 40})),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        EmptyGraph::getTestCaseName);
}  // namespace
