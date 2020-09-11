// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "execution_graph_tests/unique_node_names.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_CASE_P(NoReshape, ExecGraphUniqueNodeNames,
                        ::testing::Combine(
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::SizeVector({1, 2, 5, 5})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ExecGraphUniqueNodeNames::getTestCaseName);
}  // namespace
