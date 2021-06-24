// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/nonzero.hpp"

using namespace ngraph;
using namespace LayerTestsDefinitions;

namespace {
TEST_P(NonZeroLayerTest, Serialize) {
    Serialize();
}

std::vector<InferenceEngine::SizeVector> inputDims = {
    {7},       {1000},         {3, 5},       {65, 33},       {33, 65},
    {1, 1000}, {223, 217, 21}, {3, 4, 5, 1}, {3, 4, 1, 5, 1}};

std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::U8, InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::I32,
};

ConfigMap config;

INSTANTIATE_TEST_SUITE_P(
    smoke_NonZeroLayerTest, NonZeroLayerTest,
    ::testing::Combine(::testing::ValuesIn(inputDims),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::Values(CommonTestUtils::DEVICE_CPU),
                       ::testing::Values(config)));
}  // namespace
