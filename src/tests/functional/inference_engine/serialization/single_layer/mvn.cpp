// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/mvn.hpp"

using namespace LayerTestsDefinitions;

std::vector<InferenceEngine::Precision> dataPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

const std::vector<bool> normalizeVariance = {true, false};

// ------------------- MVN-1 -------------------------------------------------
const std::vector<std::vector<size_t>> inputShapes = {{1, 10, 5, 7, 8},
                                                      {1, 3, 8, 9, 49}};

const std::vector<ngraph::AxisSet> axes = {{1, 2, 3}, {2, 3}};
const std::vector<bool> acrossChannels = {true, false};
const std::vector<ngraph::AxisSet> emptyReductionAxes = {{}};
const std::vector<bool> emptyAcrossChannels = {{}};

const std::vector<double> epsilon = {0.000000001};

const auto MvnAcrossChannels = ::testing::Combine(
    ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(dataPrecisions),
    ::testing::ValuesIn(emptyReductionAxes), ::testing::ValuesIn(acrossChannels),
    ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

const auto MvnReductionAxes = ::testing::Combine(
    ::testing::ValuesIn(inputShapes), ::testing::ValuesIn(dataPrecisions),
    ::testing::ValuesIn(axes), ::testing::ValuesIn(emptyAcrossChannels),
    ::testing::ValuesIn(normalizeVariance), ::testing::ValuesIn(epsilon),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

TEST_P(Mvn1LayerTest, Serialize) {
    Serialize();
}

INSTANTIATE_TEST_SUITE_P(smoke_INTEL_CPU_TestsMVN_across_channels, Mvn1LayerTest, MvnAcrossChannels,
                        Mvn1LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_INTEL_CPU_TestsMVN_reduction_axes, Mvn1LayerTest, MvnReductionAxes,
                        Mvn1LayerTest::getTestCaseName);

// ------------------- MVN-6 -------------------------------------------------

std::vector<InferenceEngine::Precision> idxPrecisions = {
    InferenceEngine::Precision::I32, InferenceEngine::Precision::I64};

const std::vector<std::string> epsMode = {"inside_sqrt", "outside_sqrt"};

const std::vector<float> epsilonF = {0.0001};

TEST_P(Mvn6LayerTest, Serialize) {
    Serialize();
}

INSTANTIATE_TEST_SUITE_P(
    smoke_MVN_5D, Mvn6LayerTest,
    ::testing::Combine(::testing::ValuesIn(std::vector<std::vector<size_t>>{
                           {1, 10, 5, 7, 8}, {1, 3, 8, 9, 49}}),
                       ::testing::ValuesIn(dataPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::ValuesIn(std::vector<std::vector<int>>{
                           {1, 2, 3, 4}, {2, 3, 4}}),
                       ::testing::ValuesIn(normalizeVariance),
                       ::testing::ValuesIn(epsilonF),
                       ::testing::ValuesIn(epsMode),
                       ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    Mvn6LayerTest::getTestCaseName);
