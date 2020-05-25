// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/multiply.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace MultiplyTestDefinitions;

namespace {
std::vector<std::vector<std::vector<size_t>>> inShapes = {
        {{2}},
        {{1, 1, 1, 3}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
        {{1, 1, 1, 1, 1, 1, 3}},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}}
};

std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::FP16,
};

std::vector<MultiplicationType> multiplicationTypes = { MultiplicationType::SCALAR,
                                                        MultiplicationType::VECTOR,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "1638.4"},
    {"GNA_SCALE_FACTOR_1", "1638.4"}
};

const auto multiply_params_constant = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::Values(SecondaryInputType::CONSTANT),
    ::testing::ValuesIn(multiplicationTypes),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(CommonTestUtils::DEVICE_GNA),
    ::testing::Values(additional_config));

const auto multiply_params_parameter = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::Values(SecondaryInputType::PARAMETER),
    ::testing::ValuesIn(multiplicationTypes),
    ::testing::ValuesIn(netPrecisions),
    ::testing::Values(CommonTestUtils::DEVICE_GNA),
    ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(multilpy_constant, MultiplyLayerTest, multiply_params_constant, MultiplyLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(DISABLED_multilpy_parameter, MultiplyLayerTest, multiply_params_parameter, MultiplyLayerTest::getTestCaseName);
}  // namespace
