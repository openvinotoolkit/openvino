// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/lrn.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<std::vector<int64_t>> axes = {{1}, {2, 3}};

const double alpha = 9.9e-05;
const double beta = 2;
const double bias = 1.0;
const size_t size = 5;

INSTANTIATE_TEST_SUITE_P(smoke_LrnCheck, LrnLayerTest,
                        ::testing::Combine(::testing::Values(alpha),
                                           ::testing::Values(beta),
                                           ::testing::Values(bias),
                                           ::testing::Values(size),
                                           ::testing::ValuesIn(axes),
                                           ::testing::ValuesIn(netPrecisions),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                           ::testing::Values(std::vector<size_t>({10, 10, 3, 2})),
                                           ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        LrnLayerTest::getTestCaseName);

}  // namespace
