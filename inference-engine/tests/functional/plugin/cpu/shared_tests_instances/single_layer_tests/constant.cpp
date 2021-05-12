// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/constant.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

std::vector<std::vector<size_t>> shapes{
    {2, 2, 3},
    {3, 4, 1},
    {1, 1, 12},
};

std::vector<InferenceEngine::Precision> precisions{
    InferenceEngine::Precision::BF16, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP64,
    InferenceEngine::Precision::U4,   InferenceEngine::Precision::U8,
    InferenceEngine::Precision::U16,  InferenceEngine::Precision::U32,
    InferenceEngine::Precision::I4,   InferenceEngine::Precision::I8,
    InferenceEngine::Precision::I16,  InferenceEngine::Precision::I32,
};

std::vector<std::string> data{"0", "1", "2", "3", "4", "5", "6", "7", "0", "1", "2", "3"};

std::vector<InferenceEngine::Precision> precisionsWithNegativeValues{
    InferenceEngine::Precision::BF16, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP64,
    InferenceEngine::Precision::I4,   InferenceEngine::Precision::I8,
    InferenceEngine::Precision::I16,  InferenceEngine::Precision::I32,
};

std::vector<std::string> dataWithNegativeValues{"1", "-2", "3", "-4", "5", "-6",
                                                "7", "-1", "2", "-3", "4", "-5"};

INSTANTIATE_TEST_CASE_P(smoke_Constant, ConstantLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shapes),
                                           ::testing::ValuesIn(precisions), ::testing::Values(data),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConstantLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Constant_with_negative_values, ConstantLayerTest,
                        ::testing::Combine(::testing::ValuesIn(shapes),
                                           ::testing::ValuesIn(precisionsWithNegativeValues),
                                           ::testing::Values(dataWithNegativeValues),
                                           ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        ConstantLayerTest::getTestCaseName);
}  // namespace
