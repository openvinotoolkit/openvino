// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "execution_graph_tests/unique_node_names.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ExecutionGraphTests;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_SUITE_P(smoke_NoReshape, ExecGraphUniqueNodeNames,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::SizeVector({1, 2, 5, 5})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        ExecGraphUniqueNodeNames::getTestCaseName);
}  // namespace
