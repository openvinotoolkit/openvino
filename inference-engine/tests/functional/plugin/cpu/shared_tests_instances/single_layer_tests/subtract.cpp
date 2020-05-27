// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstddef>
#include <vector>
#include <map>

#include "single_layer_tests/subtract.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace SubtractTestDefinitions;
namespace {

std::vector<std::vector<std::vector<size_t>>> inputShapes = { {std::vector<std::size_t>({1, 30})} };

std::vector<SecondaryInputType> secondaryInputTypes = { SecondaryInputType::CONSTANT,
                                                        SecondaryInputType::PARAMETER,
};

std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::FP16,
};

std::map<std::string, std::string> additional_config = {};

const auto subtraction_params_vector_fp32 = ::testing::Combine(
                                                     ::testing::ValuesIn(inputShapes),
                                                     ::testing::ValuesIn(secondaryInputTypes),
                                                     ::testing::Values(SubtractionType::VECTOR),
                                                     ::testing::Values(InferenceEngine::Precision::FP32),
                                                     ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                     ::testing::Values(additional_config));

const auto subtraction_params_vector_fp16_parameter = ::testing::Combine(
                                                               ::testing::ValuesIn(inputShapes),
                                                               ::testing::Values(SecondaryInputType::PARAMETER),
                                                               ::testing::Values(SubtractionType::VECTOR),
                                                               ::testing::Values(InferenceEngine::Precision::FP16),
                                                               ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                               ::testing::Values(additional_config));

const auto subtraction_params_vector_fp16_constant = ::testing::Combine(
                                                              ::testing::ValuesIn(inputShapes),
                                                              ::testing::Values(SecondaryInputType::CONSTANT),
                                                              ::testing::Values(SubtractionType::VECTOR),
                                                              ::testing::Values(InferenceEngine::Precision::FP16),
                                                              ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                              ::testing::Values(additional_config));

const auto subtraction_params_scalar = ::testing::Combine(
                                                ::testing::ValuesIn(inputShapes),
                                                ::testing::ValuesIn(secondaryInputTypes),
                                                ::testing::Values(SubtractionType::SCALAR),
                                                ::testing::ValuesIn(netPrecisions),
                                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                                ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CompareWithRefs_vector_fp32, SubtractLayerTest,
                        subtraction_params_vector_fp32, SubtractLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(CompareWithRefs_vector_fp16_parameter, SubtractLayerTest,
                        subtraction_params_vector_fp16_parameter, SubtractLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_CompareWithRefs_vector_fp16_constant, SubtractLayerTest,
                        subtraction_params_vector_fp16_constant, SubtractLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_CompareWithRefs_scalar, SubtractLayerTest,
                        subtraction_params_scalar, SubtractLayerTest::getTestCaseName);
}  // namespace
