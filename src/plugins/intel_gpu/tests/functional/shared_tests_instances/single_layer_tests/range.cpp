// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <initializer_list>

#include <single_layer_tests/range.hpp>
#include <common_test_utils/test_constants.hpp>


namespace {

const std::initializer_list<float> start { 1.0, 1.2 };
const std::initializer_list<float> stop { 5.0, 5.2 };
const std::initializer_list<float> step { 1.0, 0.1 };

const std::initializer_list<InferenceEngine::Precision> netPrecisions {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

using LayerTestsDefinitions::RangeLayerTest;
INSTANTIATE_TEST_SUITE_P(smoke_Basic, RangeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(start),
                                ::testing::ValuesIn(stop),
                                ::testing::ValuesIn(step),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        RangeLayerTest::getTestCaseName);
}  // namespace
