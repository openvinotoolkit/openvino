// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/softmax.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
};

const std::vector<InferenceEngine::Layout> inputLayouts2D = {
        InferenceEngine::Layout::NC,
};

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inputStaticShape2D = {
        {{}, {{1, 100}}},
        {{}, {{100, 1}}},
        {{}, {{10, 10}}},
};

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inputDynamicShape2D = {
        {{ngraph::Dimension::dynamic(), 10}, {{1, 10}, {2, 10}, {10, 10}}},
        {{ngraph::Dimension(1, 10), 10}, {{1, 10}, {2, 10}, {10, 10}}},
        {{10, ngraph::Dimension::dynamic()}, {{10, 1}, {10, 5}, {10, 10}}},
        {{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()}, {{1, 10}, {2, 10}, {10, 10}}}
};

const std::vector<size_t> axis2D = {
        0, 1
};

const auto params2D_static = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::ValuesIn(inputLayouts2D),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputStaticShape2D),
        testing::ValuesIn(axis2D),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(std::map<std::string, std::string>())
);

const auto params2D_dynamic = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::ValuesIn(inputLayouts2D),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputDynamicShape2D),
        testing::ValuesIn(axis2D),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax2D_static,
        SoftMaxLayerTest,
        params2D_static,
        SoftMaxLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax2D_dynamic,
        SoftMaxLayerTest,
        params2D_dynamic,
        SoftMaxLayerTest::getTestCaseName
);

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inputStaticShape4D = {
        {{}, {{1, 100, 1, 1}}},
        {{}, {{50, 100, 4, 1}}},
        {{}, {{2, 100, 10, 1}}},
};

const std::vector<std::pair<ngraph::PartialShape, std::vector<ngraph::Shape>>> inputDynamicShape4D = {
        {{ngraph::Dimension::dynamic(), 100, ngraph::Dimension(1, 10), 1}, {{1, 100, 1, 1}, {100, 100, 5, 1}}},
        {{ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()},
                                                                           {{1, 100, 1, 1}, {50, 100, 4, 1}, {2, 100, 10, 1}}},
};

const std::vector<size_t> axis4D = {0, 1, 2, 3};

const auto params4Dstatic = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputStaticShape4D),
        testing::ValuesIn(axis4D),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(std::map<std::string, std::string>())
);

const auto params4Ddynamic = testing::Combine(
        testing::ValuesIn(netPrecisions),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        testing::Values(InferenceEngine::Layout::NCHW),
        testing::Values(InferenceEngine::Layout::ANY),
        testing::ValuesIn(inputDynamicShape4D),
        testing::ValuesIn(axis4D),
        testing::Values(CommonTestUtils::DEVICE_CPU),
        testing::Values(std::map<std::string, std::string>())
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax4D_static,
        SoftMaxLayerTest,
        params2D_static,
        SoftMaxLayerTest::getTestCaseName
);

INSTANTIATE_TEST_SUITE_P(
        smoke_SoftMax4D_dynamic,
        SoftMaxLayerTest,
        params2D_dynamic,
        SoftMaxLayerTest::getTestCaseName
);

}  // namespace
