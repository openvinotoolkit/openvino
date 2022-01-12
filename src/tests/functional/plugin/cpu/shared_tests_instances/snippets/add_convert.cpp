// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/add_convert.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32
    };
    INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, Add,
                         ::testing::Combine(
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::SizeVector({1, 42, 16, 64})),
                                 ::testing::Values(InferenceEngine::SizeVector({1, 42, 16,  1})),
                                 ::testing::Values(1), // one node - Add
                                 ::testing::Values(0), // SnippetsMarkSkipped disables tokenization for eltwise chains after inputs
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         AddConvert::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Snippets_Eltwise, AddConvert,
            ::testing::Combine(
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::SizeVector({1, 42, 16, 64})),
            ::testing::Values(InferenceEngine::SizeVector({1, 42, 16,  1})),
            ::testing::Values(3), // Add + 2 converts after inputs
            ::testing::Values(1), // Subgraph is created, since the inputs are followed by converts
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                             AddConvert::getTestCaseName);

}  // namespace