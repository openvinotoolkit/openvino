// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/trivial_concat.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<std::vector<size_t>> inShapesMultipleInputs = {
    {40, 40, 40},
    {5, 5},
    {32, 32, 32, 128},
    {32, 212},
    {132, 64},
    {61, 53},
    {234, 234, 222},
    {234, 234, 222, 1, 2, 3},
    {234, 234, 222, 64},
    {234, 234, 222, 33, 22},
    {234, 234, 222, 23},
    {5, 5, 5, 5, 5, 5, 5, 5},
};

static std::vector<std::vector<size_t>> getInShapes() {
    std::vector<std::vector<size_t>> shapes = {
        {1, 1024},
        {1, 1, 33, 16},
        {1, 1, 65, 16},
        {10, 16},
        {10, 64},
        {1, 1001},
    };
    for (size_t s = 1; s <= 69; s++) {
        shapes.push_back({ 1, s });
    }
    for (size_t s = 100; s <= 169; s++) {
        shapes.push_back({ 1, s });
    }
    return shapes;
}

std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

std::map<std::string, std::string> additional_config = {
    {"GNA_COMPACT_MODE", "NO"},
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "2000.0"},
};

INSTANTIATE_TEST_SUITE_P(smoke_trivial_concat_Basic, TrivialConcatLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(getInShapes()),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config)),
    TrivialConcatLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_trivial_concat_Basic_2Inputs, TrivialConcatLayerTest2Inputs,
    ::testing::Combine(
        ::testing::ValuesIn(getInShapes()),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config)),
    TrivialConcatLayerTest2Inputs::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_trivial_concat_Basic_Multi, TrivialConcatLayerTest_MultipleInputs,
    ::testing::Combine(
        ::testing::ValuesIn(inShapesMultipleInputs),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config)),
    TrivialConcatLayerTest_MultipleInputs::getTestCaseName);
}  // namespace
