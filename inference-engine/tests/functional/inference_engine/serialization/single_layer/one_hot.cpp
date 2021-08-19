// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/one_hot.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(OneHotLayerTest, Serialize) { Serialize(); }

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::I32 };

const std::vector<ngraph::element::Type> argDepthType_IC = { ngraph::element::i32 };
const std::vector<int64_t> argDepth_IC = { 5, 1017 };
const std::vector<ngraph::element::Type> argSetType_IC = { ngraph::element::i32, ngraph::element::i64 };
const std::vector<float> argOnValue_IC = { 1, -29 };
const std::vector<float> argOffValue_IC = { -1, 127 };
const std::vector<int64_t> argAxis_IC = {0, 1, -1};
const std::vector<std::vector<size_t>> inputShapes_IC = {{4, 5}, {3, 7}};

const auto oneHotParams = testing::Combine(
        testing::ValuesIn(argDepthType_IC),
        testing::ValuesIn(argDepth_IC),
        testing::ValuesIn(argSetType_IC),
        testing::ValuesIn(argOnValue_IC),
        testing::ValuesIn(argOffValue_IC),
        testing::ValuesIn(argAxis_IC),
        testing::ValuesIn(netPrecisions),
        testing::ValuesIn(inputShapes_IC),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_OneHotConstSerialization, OneHotLayerTest, oneHotParams,
                        OneHotLayerTest::getTestCaseName);
} // namespace
