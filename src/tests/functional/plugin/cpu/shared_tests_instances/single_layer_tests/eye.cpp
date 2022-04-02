// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/eye.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;


const std::vector<InferenceEngine::Precision> netPRCs = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32
};

const auto AdaPool3DCases = ::testing::Combine(
        ::testing::ValuesIn(
                std::vector<std::vector<size_t>> {
                        { 1, 2, 1},
                        { 1, 1, 3 },
                        { 3, 17, 5 }}),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1}, {3}, {5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(netPRCs),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsAdaPool3D, EyeLayerTest, AdaPool3DCases, EyeLayerTest::getTestCaseName);

const auto AdaPool4DCases = ::testing::Combine(
        ::testing::ValuesIn(
                std::vector<std::vector<size_t>> {
                        { 1, 2, 1, 2},
                        { 1, 1, 3, 2},
                        { 3, 17, 5, 1}}),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1, 1}, {3, 5}, {5, 5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(netPRCs),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsAdaPool4D, EyeLayerTest, AdaPool4DCases, EyeLayerTest::getTestCaseName);

const auto AdaPool5DCases = ::testing::Combine(
        ::testing::ValuesIn(
                std::vector<std::vector<size_t>> {
                        { 1, 2, 1, 2, 2},
                        { 1, 1, 3, 2, 3},
                        { 3, 17, 5, 1, 2}}),
        ::testing::ValuesIn(std::vector<std::vector<int>>{ {1, 1, 1}, {3, 5, 3}, {5, 5, 5} }),
        ::testing::ValuesIn(std::vector<std::string>{"max", "avg"}),
        ::testing::ValuesIn(netPRCs),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(z_smoke_TestsAdaPool5D, EyeLayerTest, AdaPool5DCases, EyeLayerTest::getTestCaseName);
