// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/pad.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<ngraph::helpers::PadMode> padMode = {
        ngraph::helpers::PadMode::EDGE,
        ngraph::helpers::PadMode::REFLECT,
        ngraph::helpers::PadMode::SYMMETRIC
};

const std::vector<float> argPadValue = {0.f, 1.f, -1.f, 2.5f};

const std::vector<std::vector<int64_t>> padsBegin4D = {{0, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 1}, {0, 3, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd4D   = {{0, 0, 0, 0}, {0, 3, 0, 0}, {0, 0, 0, 2}, {0, 3, 0, 1}};

const std::vector<std::vector<int64_t>> padsBegin3D = {{0, 0, 0}, {3, 0, 0}, {0, 0, 1}, {0, 1, 1}, {3, 0, 1}};
const std::vector<std::vector<int64_t>> padsEnd3D   = {{0, 0, 0}, {3, 0, 0}, {0, 0, 2}, {3, 0, 1}};

const auto pad4DConstParams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad4DConst,
        PadLayerTest,
        pad4DConstParams,
        PadLayerTest::getTestCaseName
);

const auto pad4DParams = testing::Combine(
        testing::ValuesIn(padsBegin4D),
        testing::ValuesIn(padsEnd4D),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{1, 5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad4D,
        PadLayerTest,
        pad4DParams,
        PadLayerTest::getTestCaseName
);

const auto pad3DConstParams = testing::Combine(
        testing::ValuesIn(padsBegin3D),
        testing::ValuesIn(padsEnd3D),
        testing::ValuesIn(argPadValue),
        testing::Values(ngraph::helpers::PadMode::CONSTANT),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad3DConst,
        PadLayerTest,
        pad3DConstParams,
        PadLayerTest::getTestCaseName
);

const auto pad3DParams = testing::Combine(
        testing::ValuesIn(padsBegin3D),
        testing::ValuesIn(padsEnd3D),
        testing::Values(0),
        testing::ValuesIn(padMode),
        testing::Values(InferenceEngine::Precision::FP16),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(std::vector<size_t>{5, 10, 11}),
        testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_Pad3D,
        PadLayerTest,
        pad3DParams,
        PadLayerTest::getTestCaseName
);

}  // namespace
