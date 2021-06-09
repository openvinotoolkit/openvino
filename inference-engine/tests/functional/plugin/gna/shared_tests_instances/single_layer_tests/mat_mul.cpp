// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/mat_mul.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<ShapeRelatedParams> shapeRelatedParams = {
        { { {5, 1}, true }, { {5, 1}, false } },
        { { {1, 5}, false }, { {1, 5}, true } },
        { { {5}, false }, { {5}, false } },
        { { {5}, true }, { {5}, true } }
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT
};

std::map<std::string, std::string> additional_config = {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}};

INSTANTIATE_TEST_CASE_P(smoke_MatMul, MatMulTest,
        ::testing::Combine(
                ::testing::ValuesIn(shapeRelatedParams),
                ::testing::ValuesIn(inputPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(secondaryInputTypes),
                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                ::testing::Values(additional_config)),
        MatMulTest::getTestCaseName);

} // namespace