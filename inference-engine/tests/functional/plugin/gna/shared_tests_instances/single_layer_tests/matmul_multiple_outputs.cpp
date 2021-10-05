// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/matmul_multiple_outputs.hpp"
#include "common_test_utils/test_constants.hpp"

namespace LayerTestsDefinitions {
namespace {
const std::vector<size_t> input_sizes = {
    25,
    30,
    50
};

const std::vector<InferenceEngine::Precision> net_precisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

std::map<std::string, std::string> additional_config = {
    {"GNA_DEVICE_MODE", "GNA_SW_EXACT"}
};
} // namespace

INSTANTIATE_TEST_SUITE_P(smoke_MatMulMultipleOutputs, MatMulMultipleOutputsTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_sizes),
                                ::testing::ValuesIn(net_precisions),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA),
                                ::testing::Values(additional_config)),
                        MatMulMultipleOutputsTest::getTestCaseName);

} // namespace LayerTestsDefinitions
