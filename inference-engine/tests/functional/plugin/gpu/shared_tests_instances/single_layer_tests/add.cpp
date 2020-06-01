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

std::vector<std::vector<std::vector<size_t>>> inputShapes = { {{1, 200}},
                                                              {{1, 2000}},
                                                              {{1, 20000}},
                                                              {{2, 200}},
                                                              {{10, 200}},
                                                              {{1, 10, 100}},
                                                              {{4, 4, 16}},
};

std::vector<SecondaryInputType> secondaryInputTypes = { SecondaryInputType::CONSTANT,
                                                        SecondaryInputType::PARAMETER,
};

std::vector<AdditionType> additionTypes = { AdditionType::SCALAR,
                                            AdditionType::VECTOR,
};

std::vector<InferenceEngine::Precision> netPrecisions = { InferenceEngine::Precision::FP32,
                                                          InferenceEngine::Precision::FP16,
};


std::map<std::string, std::string> additional_config = {};

const auto addition_params = ::testing::Combine(
                                      ::testing::ValuesIn(inputShapes),
                                      ::testing::ValuesIn(secondaryInputTypes),
                                      ::testing::ValuesIn(additionTypes),
                                      ::testing::ValuesIn(netPrecisions),
                                      ::testing::Values(CommonTestUtils::DEVICE_GPU),
                                      ::testing::Values(additional_config));

INSTANTIATE_TEST_CASE_P(CompareWithRefs, AddLayerTest, addition_params, AddLayerTest::getTestCaseName);
}  // namespace
