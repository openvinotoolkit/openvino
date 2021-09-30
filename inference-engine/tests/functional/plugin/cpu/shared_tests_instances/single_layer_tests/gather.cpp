// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/gather.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::I8
};

// Just need to check types transformation.
const std::vector<InferenceEngine::Precision> netPrecisionsTrCheck = {
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::FP16
};

const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    staticInputShapes1D = {
        {{}, {{{4}}}}
};
const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    dynamicInputShapes1D = {
        {{{ngraph::Dimension(4, 6)}, {ngraph::Dimension(1, 3)}}, {{{4}, {1}}, {{4}, {3}}}}
};

const std::vector<std::tuple<int, int>> axesBatchDims1D = {
        std::tuple<int, int>{0, 0}
};

const auto staticGather7Params1D = testing::Combine(
        testing::ValuesIn(staticInputShapes1D),
        testing::ValuesIn(axesBatchDims1D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
const auto dynamicGather7Params1D = testing::Combine(
        testing::ValuesIn(dynamicInputShapes1D),
        testing::ValuesIn(axesBatchDims1D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape1D, Gather7LayerTest, staticGather7Params1D, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape1D, Gather7LayerTest, dynamicGather7Params1D, Gather7LayerTest::getTestCaseName);

const auto typesTrfGather7Params1D = testing::Combine(
        testing::ValuesIn(staticInputShapes1D),
        testing::ValuesIn(axesBatchDims1D),
        testing::ValuesIn(netPrecisionsTrCheck),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TypesTrf, Gather7LayerTest, typesTrfGather7Params1D, Gather7LayerTest::getTestCaseName);


const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    staticInputShapes2D = {
        {{}, {{{4, 19}}}}
};
const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    dynamicInputShapes2D = {
        {{{ngraph::Dimension(4, 6), 19}, {4, ngraph::Dimension(1, 2)}}, {{{4, 19}, {1}}, {{5, 19}, {2}}}}
};

const std::vector<std::tuple<int, int>> axesBatchDims2D = {
        std::tuple<int, int>{0, 0},
        std::tuple<int, int>{1, 0},
        std::tuple<int, int>{1, 1},
        std::tuple<int, int>{-1, -1},
};

const auto staticGather7Params2D = testing::Combine(
        testing::ValuesIn(staticInputShapes2D),
        testing::ValuesIn(axesBatchDims2D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
const auto dynamicGather7Params2D = testing::Combine(
        testing::ValuesIn(dynamicInputShapes2D),
        testing::ValuesIn(axesBatchDims2D),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_StaticShape2D, Gather7LayerTest, staticGather7Params2D, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape2D, Gather7LayerTest, dynamicGather7Params2D, Gather7LayerTest::getTestCaseName);


const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    staticInputShapesBD0 = {
        {{}, {{{4, 5, 6, 7}, {4}}, {{4, 5, 6, 7}, {2, 2}}, {{4, 5, 6, 7}, {3, 2, 4}}}}
};
const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    staticInputShapesBD1 = {
        {{}, {{{4, 5, 6, 7}, {4, 2}}, {{4, 5, 6, 7}, {4, 5, 3}}, {{4, 5, 6, 7}, {4, 1, 2, 3}}}}
};
const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    staticInputShapesBD2 = {
        {{}, {{{4, 5, 6, 7}, {4, 5, 4, 3}}, {{4, 5, 6, 7}, {4, 5, 3, 2}}}}
};
const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    staticInputShapesNegativeBD = {
        {{}, {{{4, 5, 6, 7}, {4, 5, 4}}, {{4, 5, 6, 7}, {4, 5, 3}}}}
};

const std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>>
    dynamicInputShapes4D = {
        {{{ngraph::Dimension(4, 6), 5, 6, 7}, {ngraph::Dimension(2, 4), ngraph::Dimension(1, 2), ngraph::Dimension(1, 4)}},
        {{{4, 5, 6, 7}, {4, 1, 1}},
         {{5, 5, 6, 7}, {2, 2, 1}},
         {{5, 5, 6, 7}, {3, 2, 4}},
         {{6, 5, 6, 7}, {3, 2, 3}}}}
};

const std::vector<std::tuple<int, int>> axesBD0 = {
        std::tuple<int, int>{0, 0},
        std::tuple<int, int>{1, 0},
        std::tuple<int, int>{2, 0},
        std::tuple<int, int>{-1, 0},
};
const std::vector<std::tuple<int, int>> axesBD1 = {
        std::tuple<int, int>{1, 1},
        std::tuple<int, int>{2, 1},
        std::tuple<int, int>{-1, 1},
        std::tuple<int, int>{-2, 1},
};
const std::vector<std::tuple<int, int>> axesBD2 = {
        std::tuple<int, int>{2, 2},
        std::tuple<int, int>{3, -2},
        std::tuple<int, int>{-1, 2},
        std::tuple<int, int>{-1, -2},
};
const std::vector<std::tuple<int, int>> axesNegativeBD = {
        std::tuple<int, int>{0, -3},
        std::tuple<int, int>{1, -2},
        std::tuple<int, int>{2, -2},
        std::tuple<int, int>{-2, -2},
        std::tuple<int, int>{-1, -1},
        std::tuple<int, int>{-2, -1},
};

const auto staticGather7ParamsBD0 = testing::Combine(
        testing::ValuesIn(staticInputShapesBD0),
        testing::ValuesIn(axesBD0),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
const auto staticGather7ParamsBD1 = testing::Combine(
        testing::ValuesIn(staticInputShapesBD1),
        testing::ValuesIn(axesBD1),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
const auto staticGather7ParamsBD2 = testing::Combine(
        testing::ValuesIn(staticInputShapesBD2),
        testing::ValuesIn(axesBD2),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
const auto staticGather7ParamsNegativeBD = testing::Combine(
        testing::ValuesIn(staticInputShapesBD2),
        testing::ValuesIn(axesBD2),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);
const auto dynamicGather7Params4D = testing::Combine(
        testing::ValuesIn(dynamicInputShapes4D),
        testing::ValuesIn(axesBD0),
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeBD0, Gather7LayerTest, staticGather7ParamsBD0, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeBD1, Gather7LayerTest, staticGather7ParamsBD1, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeBD2, Gather7LayerTest, staticGather7ParamsBD2, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_StaticShapeNegativeBD, Gather7LayerTest, staticGather7ParamsNegativeBD, Gather7LayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_DynamicShape4D, Gather7LayerTest, dynamicGather7Params4D, Gather7LayerTest::getTestCaseName);

}  // namespace
