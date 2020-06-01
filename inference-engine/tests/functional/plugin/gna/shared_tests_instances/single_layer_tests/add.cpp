// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <vector>
#include <map>

#include "single_layer_tests/add.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace AddTestDefinitions;
namespace {

std::vector<std::vector<InferenceEngine::SizeVector>> flat_shapes = { {{1, 200}}, {{1, 2000}}, {{1, 20000}} };
std::vector<std::vector<InferenceEngine::SizeVector>> non_flat_shapes = { {{2, 200}}, {{10, 200}}, {{1, 10, 100}}, {{4, 4, 16}} };

std::vector<SecondaryInputType> secondaryInputTypes = { SecondaryInputType::CONSTANT,
                                                        SecondaryInputType::PARAMETER,
};

std::vector<AdditionType> additionTypes = { AdditionType::SCALAR,
                                            AdditionType::VECTOR,
};

std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "1638.4"},
    {"GNA_SCALE_FACTOR_1", "1638.4"}
};

const auto addition_params_flat = ::testing::Combine(
                                           ::testing::ValuesIn(flat_shapes),
                                           ::testing::ValuesIn(secondaryInputTypes),
                                           ::testing::ValuesIn(additionTypes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                           ::testing::Values(additional_config));

const auto addition_params_non_flat = ::testing::Combine(
                                               ::testing::ValuesIn(non_flat_shapes),
                                               ::testing::ValuesIn(secondaryInputTypes),
                                               ::testing::ValuesIn(additionTypes),
                                               ::testing::ValuesIn(netPrecisions),
                                               ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                               ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_flat, AddLayerTest, addition_params_flat, AddLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(CompareWithRefs_non_flat, AddLayerTest, addition_params_non_flat, AddLayerTest::getTestCaseName);
}  // namespace
