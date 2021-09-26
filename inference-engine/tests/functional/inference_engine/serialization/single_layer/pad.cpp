// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/pad.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(PadLayerTest, Serialize) { Serialize(); }

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

const std::vector<std::vector<int64_t>> padsBegin2D = {
    {0, 0}, {1, 1}, {2, 0}, {0, 3}};
const std::vector<std::vector<int64_t>> padsEnd2D = {
    {0, 0}, {1, 1}, {0, 1}, {3, 2}};
const std::vector<float> argPadValue = {0.f, 1.f, 2.f, -1.f};

const std::vector<ngraph::helpers::PadMode> padMode = {
    ngraph::helpers::PadMode::EDGE, ngraph::helpers::PadMode::REFLECT,
    ngraph::helpers::PadMode::SYMMETRIC};

const auto pad2DConstparams = testing::Combine(
    testing::ValuesIn(padsBegin2D), testing::ValuesIn(padsEnd2D),
    testing::ValuesIn(argPadValue),
    testing::Values(ngraph::helpers::PadMode::CONSTANT),
    testing::ValuesIn(netPrecisions),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Precision::UNSPECIFIED),
    testing::Values(InferenceEngine::Layout::ANY),
    testing::Values(std::vector<size_t>{13, 5}),
    testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_Pad2DConstSerialization, PadLayerTest, pad2DConstparams,
                        PadLayerTest::getTestCaseName);
} // namespace
