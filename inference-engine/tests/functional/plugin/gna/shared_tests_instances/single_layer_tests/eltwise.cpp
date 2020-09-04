// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{1, 200}},
        // TODO: Issue 32544
        // {{2}},
        // {{1, 1, 1, 3}},
        // {{1, 2, 4}},
        // {{1, 4, 4}},
        // {{1, 4, 4, 1}}
};


std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::ADD
};

std::map<std::string, std::string> additional_config = {
        {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
        {"GNA_SCALE_FACTOR_0", "1638.4"},
        {"GNA_SCALE_FACTOR_1", "1638.4"}
};

const auto multiply_params = ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(eltwiseOpTypes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CompareWithRefs, EltwiseLayerTest, multiply_params, EltwiseLayerTest::getTestCaseName);
}  // namespace