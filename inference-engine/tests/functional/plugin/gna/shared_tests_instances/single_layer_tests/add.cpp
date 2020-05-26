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

std::vector<std::vector<std::vector<size_t>>> inputShapes = { {std::vector<std::size_t>({1, 30})} };

std::vector<SecondaryInputType> secondaryInputTypes = { SecondaryInputType::CONSTANT,
                                                        SecondaryInputType::PARAMETER,
};

std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"},
    {"GNA_SCALE_FACTOR_0", "1638.4"},
    {"GNA_SCALE_FACTOR_1", "1638.4"}
};

const auto addition_params_scalar = ::testing::Combine(
                                             ::testing::ValuesIn(inputShapes),
                                             ::testing::ValuesIn(secondaryInputTypes),
                                             ::testing::Values(AdditionType::SCALAR),
                                             ::testing::ValuesIn(netPrecisions),
                                             ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                             ::testing::Values(additional_config));

const auto addition_params_vector = ::testing::Combine(
                                             ::testing::ValuesIn(inputShapes),
                                             ::testing::ValuesIn(secondaryInputTypes),
                                             ::testing::Values(AdditionType::VECTOR),
                                             ::testing::ValuesIn(netPrecisions),
                                             ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                             ::testing::Values(additional_config));


INSTANTIATE_TEST_CASE_P(CompareWithRefs_vector, AddLayerTest, addition_params_vector, AddLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(DISABLED_CompareWithRefs_scalar, AddLayerTest, addition_params_scalar, AddLayerTest::getTestCaseName);
}  // namespace
