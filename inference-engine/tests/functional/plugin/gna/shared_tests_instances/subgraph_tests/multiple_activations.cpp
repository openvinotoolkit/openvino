// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/multiple_activations.hpp"
#include "common_test_utils/test_constants.hpp"

namespace SubgraphTestsDefinitions {
namespace {
const std::vector<size_t> input_sizes = {
    25,
    30,
    50
};

// const std::vector<size_t> output_fc = {
//     50,
//     100,
//     200
// };

const std::vector<InferenceEngine::Precision> net_precisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

std::map<std::string, std::string> additional_config = {};
} // namespace

INSTANTIATE_TEST_SUITE_P(smoke_MultipleActivations, MultipleActivationsTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_sizes),
                                ::testing::ValuesIn(net_precisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::Values(additional_config)),
                        MultipleActivationsTest::getTestCaseName);

} // namespace SubgraphTestsDefinitions
