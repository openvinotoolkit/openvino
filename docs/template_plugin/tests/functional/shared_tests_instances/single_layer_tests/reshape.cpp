// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/reshape.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(ReshapeCheckDynBatch, ReshapeLayerTest,
        ::testing::Combine(
                ::testing::Values(true),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                ::testing::Values("TEMPLATE"),
                ::testing::Values(std::map<std::string, std::string>({}))),
                ReshapeLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(ReshapeCheck, ReshapeLayerTest,
        ::testing::Combine(
                ::testing::Values(true),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
                ::testing::Values(std::vector<size_t>({10, 0, 100})),
                ::testing::Values("TEMPLATE"),
                ::testing::Values(std::map<std::string, std::string>({}))),
                ReshapeLayerTest::getTestCaseName);
}  // namespace