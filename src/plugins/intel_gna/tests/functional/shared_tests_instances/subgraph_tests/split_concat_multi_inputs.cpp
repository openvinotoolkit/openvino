// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/split_concat_multi_inputs.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
std::vector<InferenceEngine::Precision> precisions = {InferenceEngine::Precision::FP32,
                                                      InferenceEngine::Precision::FP16};

std::vector<std::map<std::string, std::string>> additionalConfig = {{{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
                                                                    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}}};

std::vector<std::vector<size_t>> inputShapes = {{1, 10}, {1, 50}, {1, 32}, {1, 512}};

std::vector<size_t> splitsNum = {2, 3, 4, 10};

std::vector<bool> isFc = {false, true};

INSTANTIATE_TEST_SUITE_P(smoke_splitConcatMultiInputs,
                         SplitConcatMultiInputsTest,
                         ::testing::Combine(::testing::ValuesIn(precisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(additionalConfig),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::ValuesIn(splitsNum),
                                            ::testing::ValuesIn(isFc)),
                         SplitConcatMultiInputsTest::getTestCaseName);
}  // namespace
