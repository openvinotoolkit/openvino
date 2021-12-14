// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/pad.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<std::vector<int64_t>> padsBegin2D = {{0, 0}, {1, 1}, {2, 0}, {0, 3}};
const std::vector<std::vector<int64_t>> padsEnd2D   = {{0, 0}, {1, 1}, {0, 1}, {3, 2}};
const std::vector<float> argPadValue = {0.f, 1.f, 2.f, -1.f};

const std::vector<ngraph::helpers::PadMode> padMode = {
        ngraph::helpers::PadMode::EDGE,
        ngraph::helpers::PadMode::REFLECT,
};

const auto pad2DConstparams = testing::Combine(
        testing::ValuesIn(padsBegin2D),
        testing::ValuesIn(padsEnd2D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{13, 5}),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad2DConst,
        PadLayerTest,
        pad2DConstparams,
        PadLayerTest::getTestCaseName
);

const auto pad2Dparams = testing::Combine(
        testing::ValuesIn(padsBegin2D),
        testing::ValuesIn(padsEnd2D),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{13, 5}),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad2D,
        PadLayerTest,
        pad2Dparams,
        PadLayerTest::getTestCaseName
);

const std::vector<std::vector<int64_t>> padsBegin4D = {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 0, 1, 0}, {0, 3, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd4D   = {{0, 0, 0, 0}, {1, 1, 1, 1}, {2, 0, 0, 1}, {1, 3, 2, 0}};

const auto pad4DConstparams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad4DConst,
        PadLayerTest,
        pad4DConstparams,
        PadLayerTest::getTestCaseName
);

const auto pad4Dparams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{3, 5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad4D,
        PadLayerTest,
        pad4Dparams,
        PadLayerTest::getTestCaseName
);

}  // namespace
