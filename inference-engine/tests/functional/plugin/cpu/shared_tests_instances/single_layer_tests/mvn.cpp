// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mvn.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inputShapes = {
    {1, 32, 17},
    {1, 37, 9},
    {1, 16, 5, 8},
    {2, 19, 5, 10},
    {7, 32, 2, 8},
    {5, 8, 3, 5},
    {4, 41, 6, 9},
    {1, 32, 8, 1, 6},
    {1, 9, 1, 15, 9},
    {6, 64, 6, 1, 18},
    {2, 31, 2, 9, 1},
    {10, 16, 5, 10, 6}
};

const std::vector<bool> acrossChannels = {
    true,
    false
};

const std::vector<bool> normalizeVariance = {
    true,
    false
};

const std::vector<double> epsilon = {
    0.000000001
};

const auto MvnCases = ::testing::Combine(
    ::testing::ValuesIn(inputShapes),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::ValuesIn(acrossChannels),
    ::testing::ValuesIn(normalizeVariance),
    ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_MKLDNN_TestsMVN, MvnLayerTest, MvnCases, MvnLayerTest::getTestCaseName);

